"""
EPM Dataset Loaders, Download & Cache
=====================================

Turns a :class:`~src.data.registry.DatasetSpec` into an in-memory
:class:`LoadedDataset`.

Reliability & security
----------------------
* Downloads use HTTPS only, from an explicit host allow-list, with a timeout and
  a hard byte cap. Data are written to ``*.part`` and atomically renamed, so an
  interrupted download never leaves a half-file that looks valid.
* Zip archives are extracted member-by-member: only the expected member is
  read, path-traversal names are rejected and the uncompressed size is capped.
* A ``manifest.json`` per dataset records SHA-256, size, source URL and time of
  download, plus the last error if a download failed.
* Failures raise :class:`DatasetLoadError` (never a crash, never a silent
  substitute). Set ``EPM_OFFLINE=1`` to forbid all network access.
* An in-process LRU cache keeps recently used datasets in memory so repeated
  experiments do not re-parse files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.registry import DatasetSpec, get_dataset_spec, DATASETS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_HOME = PROJECT_ROOT / "data"

ALLOWED_HOSTS = ("archive.ics.uci.edu", "raw.githubusercontent.com")
DEFAULT_TIMEOUT_S = 60
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB hard cap per file
MAX_UNZIPPED_BYTES = 400 * 1024 * 1024


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be downloaded, parsed or validated."""


@dataclass
class LoadedDataset:
    """A dataset materialised in memory.

    ``features`` never contains the target column. ``target`` holds the
    supervised target, or - for clustering/anomaly datasets - labels that may
    only be used for *evaluation* (see ``label_role``).
    """

    spec: DatasetSpec
    features: pd.DataFrame
    target: Optional[pd.Series]
    numeric_features: List[str]
    categorical_features: List[str]
    label_role: str
    time_index: Optional[pd.Index] = None
    sensitive: Optional[pd.DataFrame] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(len(self.features))

    @property
    def memory_mb(self) -> float:
        mem = self.features.memory_usage(deep=True).sum()
        if self.target is not None:
            mem += self.target.memory_usage(deep=True)
        return float(mem) / (1024 * 1024)


# =============================================================================
# Paths, manifest & status
# =============================================================================

def data_home() -> Path:
    return Path(os.environ.get("EPM_DATA_HOME", str(DEFAULT_DATA_HOME)))


def raw_dir(dataset_id: str) -> Path:
    return data_home() / "raw" / dataset_id


def processed_dir() -> Path:
    return data_home() / "processed"


def sklearn_home() -> Path:
    return data_home() / "raw" / "sklearn"


def offline_mode() -> bool:
    return os.environ.get("EPM_OFFLINE", "").strip() in ("1", "true", "yes")


def _manifest_path(dataset_id: str) -> Path:
    return raw_dir(dataset_id) / "manifest.json"


def read_manifest(dataset_id: str) -> Dict[str, Any]:
    try:
        return json.loads(_manifest_path(dataset_id).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(dataset_id: str, **updates: Any) -> None:
    d = raw_dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    m = read_manifest(dataset_id)
    m.update(updates)
    tmp = _manifest_path(dataset_id).with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, _manifest_path(dataset_id))


def _sklearn_cached(spec: DatasetSpec) -> bool:
    home = sklearn_home()
    if not home.exists():
        return False
    if spec.download_method == "sklearn_fetcher":
        return any(home.glob("cal_housing*"))
    if spec.download_method == "openml":
        return bool(read_manifest(spec.id).get("downloaded_at"))
    return False


def dataset_status(dataset_id: str) -> Dict[str, Any]:
    """Download/cache/load status for the catalogue (never triggers a download)."""
    spec = get_dataset_spec(dataset_id)
    m = read_manifest(dataset_id)
    if spec.download_method in ("bundled", "synthetic", "upload"):
        state = "available"
    elif spec.download_method in ("uci_zip", "github_raw"):
        state = "cached" if m.get("sha256") and (raw_dir(dataset_id) / m.get("file", "")).exists() else "not_downloaded"
    else:  # sklearn_fetcher / openml
        state = "cached" if _sklearn_cached(spec) else "not_downloaded"
    if state == "not_downloaded" and m.get("last_error"):
        state = "error"
    return {
        "dataset_id": dataset_id,
        "state": state,
        "in_memory": dataset_id in _MEMORY_CACHE,
        "requires_network": spec.requires_network,
        "last_error": m.get("last_error"),
        "downloaded_at": m.get("downloaded_at"),
        "sha256": m.get("sha256"),
        "bytes": m.get("bytes"),
    }


# =============================================================================
# Safe download primitives
# =============================================================================

def _check_url(url: str) -> None:
    from urllib.parse import urlparse

    p = urlparse(url)
    if p.scheme != "https":
        raise DatasetLoadError(f"Refusing non-HTTPS download URL: {url}")
    if p.hostname not in ALLOWED_HOSTS:
        raise DatasetLoadError(f"Host '{p.hostname}' is not in the download allow-list {ALLOWED_HOSTS}")


def download_file(
    url: str,
    dest: Path,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Stream ``url`` to ``dest`` atomically. Returns {'sha256', 'bytes'}."""
    if offline_mode():
        raise DatasetLoadError("EPM_OFFLINE is set: network downloads are disabled.")
    _check_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": "EPM-dataset-loader/1.0"})
    h = hashlib.sha256()
    total = 0
    try:
        with opener(req, timeout=timeout) as resp, open(part, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise DatasetLoadError(f"Download exceeded the {max_bytes // (1024*1024)} MB cap: {url}")
                h.update(chunk)
                out.write(chunk)
        if total == 0:
            raise DatasetLoadError(f"Empty response from {url}")
        os.replace(part, dest)
    except DatasetLoadError:
        part.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        part.unlink(missing_ok=True)
        raise DatasetLoadError(f"Download failed for {url}: {exc}") from exc
    return {"sha256": h.hexdigest(), "bytes": total}


def extract_member(zip_path: Path, member_suffix: str, dest: Path, max_bytes: int = MAX_UNZIPPED_BYTES) -> Path:
    """Extract the single archive member ending with ``member_suffix`` safely."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [
                i for i in zf.infolist()
                if i.filename.endswith(member_suffix) and not i.is_dir()
                and "__MACOSX" not in i.filename
            ]
            if not candidates:
                raise DatasetLoadError(f"'{member_suffix}' not found in {zip_path.name}: {zf.namelist()}")
            info = candidates[0]
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                raise DatasetLoadError(f"Unsafe path in archive: {info.filename}")
            if info.file_size > max_bytes:
                raise DatasetLoadError(f"Archive member too large ({info.file_size} bytes)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            with zf.open(info) as src, open(part, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 16)
            os.replace(part, dest)
            return dest
    except zipfile.BadZipFile as exc:
        raise DatasetLoadError(f"Corrupt archive {zip_path.name}: {exc}") from exc


def _ensure_remote_file(spec: DatasetSpec, filename: str, allow_download: bool) -> Path:
    """Return the local path of ``spec.download_url`` (download once, then cache)."""
    target = raw_dir(spec.id) / filename
    m = read_manifest(spec.id)
    if target.exists() and m.get("sha256"):
        return target
    if not allow_download:
        raise DatasetLoadError(
            f"Dataset '{spec.id}' is not cached locally and downloading is disabled. "
            f"Download it first (source: {spec.download_url})."
        )
    try:
        meta = download_file(spec.download_url, target)
    except DatasetLoadError as exc:
        _write_manifest(spec.id, last_error=str(exc), last_attempt=_now())
        raise
    _write_manifest(
        spec.id, file=filename, url=spec.download_url, downloaded_at=_now(),
        last_error=None, **meta,
    )
    return target


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# Per-dataset parsers
# =============================================================================

def _from_frame(spec: DatasetSpec, df: pd.DataFrame, target_col: Optional[str], *,
                time_index: Optional[pd.Index] = None, info: Optional[dict] = None) -> LoadedDataset:
    target = None
    if target_col is not None:
        target = df[target_col].reset_index(drop=True)
        target.name = spec.target or target_col
        df = df.drop(columns=[target_col])
    drop = [c for c in spec.dropped_columns if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    df = df.reset_index(drop=True)
    num = [c for c in df.columns if c not in spec.categorical_features and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c not in num]
    sens = None
    if spec.sensitive_features:
        cols = [c for c in spec.sensitive_features if c in df.columns]
        sens = df[cols].astype("string").fillna("missing") if cols else None
    return LoadedDataset(
        spec=spec, features=df, target=target, numeric_features=num, categorical_features=cat,
        label_role=spec.label_role, time_index=time_index, sensitive=sens, info=info or {},
    )


def _load_bundled(spec: DatasetSpec, **_: Any) -> LoadedDataset:
    from sklearn import datasets as skd

    loader = {
        "iris": skd.load_iris, "wine": skd.load_wine, "breast_cancer": skd.load_breast_cancer,
        "digits": skd.load_digits, "diabetes": skd.load_diabetes,
    }[spec.id]
    b = loader(as_frame=True)
    df = b.frame.copy()
    tcol = b.target.name
    if spec.id == "iris":
        df.columns = list(spec.numeric_features) + [tcol]
        df[tcol] = df[tcol].map(dict(enumerate(b.target_names)))
    elif spec.id == "wine":
        df[tcol] = df[tcol].map(lambda i: f"class_{i}")
    elif spec.id == "breast_cancer":
        df[tcol] = (df[tcol] == 0).astype(int)  # malignant (sklearn 0) -> positive class 1
    return _from_frame(spec, df, tcol)


def _load_california(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    from sklearn.datasets import fetch_california_housing

    if offline_mode():
        allow_download = False
    try:
        b = fetch_california_housing(data_home=str(sklearn_home()), as_frame=True,
                                     download_if_missing=allow_download)
    except OSError as exc:
        raise DatasetLoadError(
            f"California Housing not cached and download {'failed' if allow_download else 'disabled'}: {exc}"
        ) from exc
    except Exception as exc:  # network errors surface as URLError/HTTPError
        _write_manifest(spec.id, last_error=str(exc), last_attempt=_now())
        raise DatasetLoadError(f"California Housing download failed: {exc}") from exc
    _write_manifest(spec.id, downloaded_at=read_manifest(spec.id).get("downloaded_at") or _now(), last_error=None)
    return _from_frame(spec, b.frame.copy(), "MedHouseVal")


_BANK_COLUMNS = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
                 "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]


def _fetch_openml(spec: DatasetSpec, data_id: int, allow_download: bool) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    if offline_mode():
        allow_download = False
    if not allow_download and not read_manifest(spec.id).get("downloaded_at"):
        raise DatasetLoadError(f"Dataset '{spec.id}' (OpenML {data_id}) is not cached and downloading is disabled.")
    try:
        b = fetch_openml(data_id=data_id, as_frame=True, data_home=str(sklearn_home()), parser="auto")
    except Exception as exc:
        _write_manifest(spec.id, last_error=str(exc), last_attempt=_now())
        raise DatasetLoadError(f"OpenML download/parse failed for data_id={data_id}: {exc}") from exc
    _write_manifest(spec.id, downloaded_at=read_manifest(spec.id).get("downloaded_at") or _now(),
                    url=spec.download_url, last_error=None)
    return b.frame.copy()


def _load_adult(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    df = _fetch_openml(spec, 1590, allow_download)
    if "class" not in df.columns:
        raise DatasetLoadError(f"Unexpected Adult schema: {list(df.columns)}")
    y = df.pop("class").astype(str).str.strip().str.rstrip(".")
    df["income_gt_50k"] = (y == ">50K").astype(int)
    for c in spec.categorical_features:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return _from_frame(spec, df, "income_gt_50k")


def _load_bank(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    df = _fetch_openml(spec, 1461, allow_download)
    vcols = [f"V{i}" for i in range(1, 17)]
    if not all(c in df.columns for c in vcols) or "Class" not in df.columns:
        raise DatasetLoadError(f"Unexpected Bank Marketing schema: {list(df.columns)}")
    df = df.rename(columns=dict(zip(vcols, _BANK_COLUMNS)))
    df["subscribed"] = (df.pop("Class").astype(str) == "2").astype(int)
    for c in spec.categorical_features:
        df[c] = df[c].astype("object")
    return _from_frame(spec, df, "subscribed")


def _load_creditcard(spec: DatasetSpec, allow_download: bool = True, allow_large: bool = False, **_: Any) -> LoadedDataset:
    if not allow_large:
        raise DatasetLoadError(
            "credit_card_fraud is a large dataset (~150 MB download, ~285k rows). "
            "Pass allow_large=True (UI: 'Allow large datasets') to load it."
        )
    df = _fetch_openml(spec, 1597, allow_download)
    if "Class" not in df.columns:
        raise DatasetLoadError(f"Unexpected creditcard schema: {list(df.columns)}")
    df["is_fraud"] = (df.pop("Class").astype(str).str.strip("'\" ") == "1").astype(int)
    return _from_frame(spec, df, "is_fraud")


def _load_energy(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    zpath = _ensure_remote_file(spec, "energy_efficiency.zip", allow_download)
    xlsx = extract_member(zpath, "ENB2012_data.xlsx", raw_dir(spec.id) / "ENB2012_data.xlsx")
    try:
        df = pd.read_excel(xlsx)
    except ImportError as exc:
        raise DatasetLoadError(
            "Reading the Energy Efficiency .xlsx needs the optional 'openpyxl' package "
            "(pip install openpyxl)."
        ) from exc
    df = df.dropna(how="all").dropna(axis=1, how="all")
    expected = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
    if list(df.columns[:10]) != expected:
        raise DatasetLoadError(f"Unexpected Energy Efficiency columns: {list(df.columns)}")
    df = df[expected]
    df.columns = list(spec.numeric_features) + ["heating_load", "cooling_load"]
    return _from_frame(spec, df, "heating_load")


def _load_wholesale(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    zpath = _ensure_remote_file(spec, "wholesale_customers.zip", allow_download)
    csv = extract_member(zpath, "Wholesale customers data.csv", raw_dir(spec.id) / "wholesale.csv")
    df = pd.read_csv(csv)
    need = ["Channel", "Region", *spec.numeric_features]
    if any(c not in df.columns for c in need):
        raise DatasetLoadError(f"Unexpected Wholesale columns: {list(df.columns)}")
    df = df[need].drop(columns=["Region"])
    df["Channel"] = df["Channel"].map({1: "Horeca", 2: "Retail"})
    return _from_frame(spec, df, "Channel")


def _load_seeds(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    zpath = _ensure_remote_file(spec, "seeds.zip", allow_download)
    txt = extract_member(zpath, "seeds_dataset.txt", raw_dir(spec.id) / "seeds_dataset.txt")
    df = pd.read_csv(txt, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != 8:
        raise DatasetLoadError(f"Seeds file should have 8 columns, found {df.shape[1]}")
    df.columns = list(spec.numeric_features) + ["variety"]
    df["variety"] = df["variety"].astype(int).map({1: "Kama", 2: "Rosa", 3: "Canadian"})
    return _from_frame(spec, df, "variety")


def _load_air_quality(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    zpath = _ensure_remote_file(spec, "air_quality.zip", allow_download)
    csv = extract_member(zpath, "AirQualityUCI.csv", raw_dir(spec.id) / "AirQualityUCI.csv")
    df = pd.read_csv(csv, sep=";", decimal=",")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].dropna(how="all")
    if "Date" not in df.columns or "Time" not in df.columns:
        raise DatasetLoadError(f"Unexpected Air Quality columns: {list(df.columns)}")
    ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str).str.replace(".", ":", regex=False),
                        format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.drop(columns=["Date", "Time"])
    df = df.replace(-200, np.nan)
    df.index = ts
    df = df[~df.index.isna()].sort_index()
    missing = [c for c in spec.numeric_features if c not in df.columns]
    if missing:
        raise DatasetLoadError(f"Air Quality missing columns {missing}")
    df = df[list(spec.numeric_features)].astype(float)
    return _series_dataset(spec, df)


def _load_household_power(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    cache = processed_dir() / "household_power_hourly.csv.gz"
    if not cache.exists():
        zpath = _ensure_remote_file(spec, "household_power.zip", allow_download)
        txt = extract_member(zpath, "household_power_consumption.txt", raw_dir(spec.id) / "household_power_consumption.txt")
        raw = pd.read_csv(txt, sep=";", na_values=["?", ""], low_memory=False)
        if "Date" not in raw.columns or "Time" not in raw.columns:
            raise DatasetLoadError(f"Unexpected Household Power columns: {list(raw.columns)}")
        ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        raw = raw.drop(columns=["Date", "Time"]).astype(float)
        raw.index = ts
        hourly = raw[~raw.index.isna()].resample("1h").mean()
        processed_dir().mkdir(parents=True, exist_ok=True)
        part = cache.with_name(cache.name + ".part")
        hourly.to_csv(part, index_label="datetime", compression="gzip")
        os.replace(part, cache)
        # Free the 127 MB text file; the zip stays cached for re-derivation.
        (raw_dir(spec.id) / "household_power_consumption.txt").unlink(missing_ok=True)
    df = pd.read_csv(cache, index_col="datetime", parse_dates=["datetime"])
    df = df[list(spec.numeric_features)]
    return _series_dataset(spec, df)


def _series_dataset(spec: DatasetSpec, df: pd.DataFrame) -> LoadedDataset:
    """Multivariate series: features include the target channel (past values are inputs)."""
    time_index = df.index
    target = df[spec.target].reset_index(drop=True)
    feats = df.reset_index(drop=True)
    info = {"start": str(time_index[0]), "end": str(time_index[-1])} if len(time_index) else {}
    return LoadedDataset(
        spec=spec, features=feats, target=target, numeric_features=list(feats.columns),
        categorical_features=[], label_role="target", time_index=time_index, info=info,
    )


def _load_epm_telemetry(spec: DatasetSpec, seed: int = 42, **_: Any) -> LoadedDataset:
    from src.timeseries.dataset import generate_multivariate_sensor_telemetry

    df, _names = generate_multivariate_sensor_telemetry(n_timesteps=spec.n_samples or 2000, random_seed=seed)
    df.index = pd.RangeIndex(len(df), name="step")
    return _series_dataset(spec, df)


def _load_epm_tabular(spec: DatasetSpec, seed: int = 42, **_: Any) -> LoadedDataset:
    from src.classical_ml.baselines import generate_sensor_regression_dataset, generate_sensor_failure_dataset

    gen = generate_sensor_regression_dataset if spec.id == "epm_sensor_regression" else generate_sensor_failure_dataset
    X, y, names = gen(n_samples=spec.n_samples or 1200, random_seed=seed)
    df = pd.DataFrame(X, columns=names)
    df[spec.target] = y
    return _from_frame(spec, df, spec.target)


def _load_blobs(spec: DatasetSpec, seed: int = 42, **_: Any) -> LoadedDataset:
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=spec.n_samples, n_features=spec.n_features, centers=spec.n_clusters_hint,
                      cluster_std=1.2, random_state=seed)
    df = pd.DataFrame(X, columns=list(spec.numeric_features))
    df[spec.target] = y
    return _from_frame(spec, df, spec.target)


def make_synthetic_anomalies(n_samples: int = 2000, n_features: int = 8, contamination: float = 0.05,
                             seed: int = 42) -> pd.DataFrame:
    """Gaussian inlier clusters + uniform outliers at an exact, controllable rate."""
    if not 0.0 < contamination < 0.5:
        raise ValueError("contamination must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    n_out = int(round(n_samples * contamination))
    n_in = n_samples - n_out
    centers = rng.uniform(-4, 4, size=(3, n_features))
    which = rng.integers(0, 3, size=n_in)
    inliers = centers[which] + rng.normal(0, 0.7, size=(n_in, n_features))
    outliers = rng.uniform(-9, 9, size=(n_out, n_features))
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(n_in, int), np.ones(n_out, int)]
    perm = rng.permutation(n_samples)
    df = pd.DataFrame(X[perm], columns=[f"x{i}" for i in range(n_features)])
    df["is_anomaly"] = y[perm]
    return df


def _load_synthetic_anomalies(spec: DatasetSpec, seed: int = 42, contamination: Optional[float] = None, **_: Any) -> LoadedDataset:
    df = make_synthetic_anomalies(spec.n_samples or 2000, spec.n_features or 8,
                                  contamination or spec.contamination_prior or 0.05, seed)
    return _from_frame(spec, df, "is_anomaly")


_NAB_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
_NAB_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def _load_nab(spec: DatasetSpec, allow_download: bool = True, **_: Any) -> LoadedDataset:
    csv = _ensure_remote_file(spec, "machine_temperature_system_failure.csv", allow_download)
    labels_path = raw_dir(spec.id) / "combined_windows.json"
    if not labels_path.exists():
        if not allow_download:
            raise DatasetLoadError("NAB label windows not cached and downloading is disabled.")
        download_file(_NAB_LABELS_URL, labels_path)
    df = pd.read_csv(csv)
    if list(df.columns) != ["timestamp", "value"]:
        raise DatasetLoadError(f"Unexpected NAB columns: {list(df.columns)}")
    ts = pd.to_datetime(df["timestamp"])
    windows = json.loads(labels_path.read_text(encoding="utf-8")).get(_NAB_KEY)
    if not windows:
        raise DatasetLoadError(f"No label windows for {_NAB_KEY} in combined_windows.json")
    flag = np.zeros(len(df), dtype=int)
    for start, end in windows:
        flag[(ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))] = 1
    out = pd.DataFrame({"value": df["value"].astype(float), "is_anomaly_window": flag})
    ld = _from_frame(spec, out, "is_anomaly_window", time_index=pd.Index(ts),
                     info={"label_windows": windows, "start": str(ts.iloc[0]), "end": str(ts.iloc[-1])})
    return ld


_LOADERS: Dict[str, Callable[..., LoadedDataset]] = {
    "iris": _load_bundled, "wine": _load_bundled, "breast_cancer": _load_bundled,
    "digits": _load_bundled, "diabetes": _load_bundled,
    "california_housing": _load_california,
    "adult": _load_adult, "bank_marketing": _load_bank, "credit_card_fraud": _load_creditcard,
    "energy_efficiency": _load_energy, "wholesale_customers": _load_wholesale, "seeds": _load_seeds,
    "air_quality": _load_air_quality, "household_power": _load_household_power,
    "epm_sensor_telemetry": _load_epm_telemetry,
    "epm_sensor_regression": _load_epm_tabular, "epm_sensor_failure": _load_epm_tabular,
    "synthetic_blobs": _load_blobs, "synthetic_anomalies": _load_synthetic_anomalies,
    "nab_machine_temperature": _load_nab,
}

assert set(_LOADERS) == set(DATASETS), "every registered dataset needs a loader"


# =============================================================================
# Public API with in-memory LRU cache
# =============================================================================

_MEMORY_CACHE: "OrderedDict[str, LoadedDataset]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
MEMORY_CACHE_MAX_ITEMS = 6
MEMORY_CACHE_MAX_MB = 600.0


def _cache_key(dataset_id: str, kwargs: Dict[str, Any]) -> str:
    extra = {k: v for k, v in sorted(kwargs.items()) if k in ("seed", "contamination")}
    return dataset_id + (json.dumps(extra) if extra else "")


def load_dataset(
    dataset_id: str,
    *,
    allow_download: bool = True,
    allow_large: bool = False,
    use_memory_cache: bool = True,
    validate: bool = True,
    **kwargs: Any,
) -> LoadedDataset:
    """Load a registered dataset (download + cache on first use).

    Raises
    ------
    KeyError            unknown dataset id
    DatasetLoadError    download/parse/validation failure (with a readable reason)
    """
    spec = get_dataset_spec(dataset_id)
    key = _cache_key(dataset_id, kwargs)
    if use_memory_cache:
        with _CACHE_LOCK:
            if key in _MEMORY_CACHE:
                _MEMORY_CACHE.move_to_end(key)
                return _MEMORY_CACHE[key]

    t0 = time.perf_counter()
    try:
        if dataset_id.startswith("upload_"):
            from src.data.uploads import load_upload

            ds = load_upload(dataset_id)
        else:
            loader = _LOADERS[dataset_id]
            ds = loader(spec, allow_download=allow_download, allow_large=allow_large, **kwargs)
    except DatasetLoadError:
        raise
    except KeyError:
        raise
    except Exception as exc:  # parsing errors from pandas etc.
        raise DatasetLoadError(f"Failed to parse dataset '{dataset_id}': {type(exc).__name__}: {exc}") from exc

    if validate:
        from src.data.validation import validate_loaded_dataset

        problems = validate_loaded_dataset(ds)
        if problems:
            raise DatasetLoadError(f"Schema validation failed for '{dataset_id}': " + "; ".join(problems))

    ds.info.setdefault("load_seconds", round(time.perf_counter() - t0, 3))
    ds.info.setdefault("memory_mb", round(ds.memory_mb, 3))

    if use_memory_cache:
        with _CACHE_LOCK:
            _MEMORY_CACHE[key] = ds
            _MEMORY_CACHE.move_to_end(key)
            while len(_MEMORY_CACHE) > MEMORY_CACHE_MAX_ITEMS or (
                len(_MEMORY_CACHE) > 1 and sum(d.memory_mb for d in _MEMORY_CACHE.values()) > MEMORY_CACHE_MAX_MB
            ):
                _MEMORY_CACHE.popitem(last=False)
    return ds


def clear_memory_cache() -> None:
    with _CACHE_LOCK:
        _MEMORY_CACHE.clear()


def download_dataset(dataset_id: str, allow_large: bool = False) -> Dict[str, Any]:
    """Explicitly download + parse + cache a dataset. Returns its status."""
    load_dataset(dataset_id, allow_download=True, allow_large=allow_large)
    return dataset_status(dataset_id)
