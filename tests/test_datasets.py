"""
Tests for the dataset registry, loaders, download/caching safety and schema /
upload validation (src/data/).

Network is never used: remote formats are exercised with small fixture files
written in the documented layout of each source, and downloads are tested with
a fake opener.
"""

import io
import json
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import loaders as L
from src.data.registry import DATASETS, TASKS, get_dataset_spec, list_datasets, validate_registry
from src.data.validation import UploadValidationError, validate_uploaded_csv


@pytest.fixture(autouse=True)
def isolated_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_DATA_HOME", str(tmp_path / "data"))
    L.clear_memory_cache()
    yield
    L.clear_memory_cache()


# ----------------------------------------------------------------- registry
class TestRegistry:
    def test_registry_is_consistent(self):
        assert validate_registry() == []

    def test_every_dataset_has_a_loader(self):
        assert set(L._LOADERS) == set(DATASETS)

    def test_every_task_has_datasets(self):
        for t in TASKS:
            assert list_datasets(t), f"no dataset for {t}"

    def test_remote_sources_are_https_and_documented(self):
        for s in DATASETS.values():
            assert s.license and s.source and s.source_url
            if s.download_url:
                assert s.download_url.startswith("https://")

    def test_unknown_dataset_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_dataset_spec("not_a_dataset")

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError):
            list_datasets("telepathy")

    def test_spec_serialises_to_json(self):
        json.dumps([s.to_dict() for s in DATASETS.values()])


# ----------------------------------------------------------------- offline loaders
OFFLINE_IDS = ["iris", "wine", "breast_cancer", "digits", "diabetes", "synthetic_blobs",
               "synthetic_anomalies", "epm_sensor_regression", "epm_sensor_failure", "epm_sensor_telemetry"]


class TestOfflineLoaders:
    @pytest.mark.parametrize("dataset_id", OFFLINE_IDS)
    def test_loads_and_validates(self, dataset_id):
        ds = L.load_dataset(dataset_id)
        spec = ds.spec
        assert ds.n_samples >= 100
        if spec.n_samples:
            assert abs(ds.n_samples - spec.n_samples) <= 2
        if spec.target and "forecasting" not in spec.tasks:
            assert spec.target not in ds.features.columns  # target never leaks into features
        assert L.dataset_status(dataset_id)["state"] == "available"

    def test_breast_cancer_positive_class_is_malignant(self):
        ds = L.load_dataset("breast_cancer")
        assert ds.target.sum() == 212  # 212 malignant cases in WDBC

    def test_memory_cache_reuses_object(self):
        a = L.load_dataset("iris")
        b = L.load_dataset("iris")
        assert a is b
        c = L.load_dataset("iris", use_memory_cache=False)
        assert c is not a

    def test_synthetic_anomaly_rate_is_controllable(self):
        df = L.make_synthetic_anomalies(1000, 4, contamination=0.08, seed=1)
        assert df["is_anomaly"].mean() == pytest.approx(0.08)
        with pytest.raises(ValueError):
            L.make_synthetic_anomalies(100, 2, contamination=0.7)

    def test_large_dataset_requires_opt_in(self):
        with pytest.raises(L.DatasetLoadError, match="large"):
            L.load_dataset("credit_card_fraud", allow_download=False)


# ----------------------------------------------------------------- downloads
class _FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class TestDownloads:
    def test_rejects_non_https(self, tmp_path):
        with pytest.raises(L.DatasetLoadError, match="non-HTTPS"):
            L.download_file("http://archive.ics.uci.edu/x.zip", tmp_path / "x")

    def test_rejects_unlisted_host(self, tmp_path):
        with pytest.raises(L.DatasetLoadError, match="allow-list"):
            L.download_file("https://evil.example.com/x.zip", tmp_path / "x")

    def test_network_error_is_wrapped_and_leaves_no_partial_file(self, tmp_path):
        def opener(req, timeout):
            raise urllib.error.URLError("unreachable")

        dest = tmp_path / "f.zip"
        with pytest.raises(L.DatasetLoadError, match="Download failed"):
            L.download_file("https://archive.ics.uci.edu/static/public/1/x.zip", dest, opener=opener)
        assert not dest.exists() and not (tmp_path / "f.zip.part").exists()

    def test_size_cap(self, tmp_path):
        dest = tmp_path / "big"
        with pytest.raises(L.DatasetLoadError, match="cap"):
            L.download_file("https://archive.ics.uci.edu/x", dest, max_bytes=10,
                            opener=lambda req, timeout: _FakeResp(b"x" * 100))
        assert not dest.exists()

    def test_empty_response_rejected(self, tmp_path):
        with pytest.raises(L.DatasetLoadError, match="Empty"):
            L.download_file("https://archive.ics.uci.edu/x", tmp_path / "e", opener=lambda r, timeout: _FakeResp(b""))

    def test_successful_download_records_hash(self, tmp_path):
        meta = L.download_file("https://raw.githubusercontent.com/a/b", tmp_path / "ok",
                               opener=lambda r, timeout: _FakeResp(b"hello"))
        assert meta["bytes"] == 5 and len(meta["sha256"]) == 64
        assert (tmp_path / "ok").read_bytes() == b"hello"

    def test_offline_mode_blocks_downloads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EPM_OFFLINE", "1")
        with pytest.raises(L.DatasetLoadError, match="OFFLINE"):
            L.download_file("https://archive.ics.uci.edu/x", tmp_path / "x")

    def test_uncached_remote_dataset_with_downloads_disabled(self):
        with pytest.raises(L.DatasetLoadError, match="not cached"):
            L.load_dataset("seeds", allow_download=False)
        assert L.dataset_status("seeds")["state"] == "not_downloaded"

    def test_failed_download_is_recorded_in_status(self, monkeypatch):
        def boom(url, dest, **kw):
            raise L.DatasetLoadError("Download failed: simulated outage")

        monkeypatch.setattr(L, "download_file", boom)
        with pytest.raises(L.DatasetLoadError):
            L.load_dataset("wholesale_customers")
        st = L.dataset_status("wholesale_customers")
        assert st["state"] == "error" and "simulated outage" in st["last_error"]


# ----------------------------------------------------------------- archive safety
def _zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestArchives:
    def test_path_traversal_rejected(self, tmp_path):
        z = tmp_path / "a.zip"
        z.write_bytes(_zip_bytes({"../../evil.csv": "x"}))
        with pytest.raises(L.DatasetLoadError, match="Unsafe"):
            L.extract_member(z, "evil.csv", tmp_path / "out.csv")

    def test_missing_member(self, tmp_path):
        z = tmp_path / "a.zip"
        z.write_bytes(_zip_bytes({"other.csv": "x"}))
        with pytest.raises(L.DatasetLoadError, match="not found"):
            L.extract_member(z, "wanted.csv", tmp_path / "o")

    def test_corrupt_archive(self, tmp_path):
        z = tmp_path / "a.zip"
        z.write_bytes(b"not a zip")
        with pytest.raises(L.DatasetLoadError, match="Corrupt"):
            L.extract_member(z, "x", tmp_path / "o")


# ----------------------------------------------------------------- source-format parsers (fixtures)
def _install_fixture(dataset_id, filename, payload):
    """Place a fixture archive where the loader expects its cached download."""
    d = L.raw_dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(payload)
    L._write_manifest(dataset_id, file=filename, sha256="fixture", downloaded_at="fixture")


class TestSourceParsers:
    def test_seeds_whitespace_format(self):
        rng = np.random.default_rng(0)
        lines = []
        for i in range(60):
            vals = rng.uniform(1, 20, 7)
            sep = "\t\t" if i % 7 == 0 else "\t"  # the real file has irregular tabs
            lines.append(sep.join(f"{v:.4f}" for v in vals) + f"\t{i % 3 + 1}")
        _install_fixture("seeds", "seeds.zip", _zip_bytes({"seeds_dataset.txt": "\n".join(lines)}))
        ds = L.load_dataset("seeds", allow_download=False)
        assert ds.features.shape == (60, 7)
        assert set(ds.target) == {"Kama", "Rosa", "Canadian"}

    def test_wholesale_format(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.integers(1, 10000, (40, 6)),
                          columns=["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"])
        df.insert(0, "Region", rng.integers(1, 4, 40))
        df.insert(0, "Channel", rng.integers(1, 3, 40))
        _install_fixture("wholesale_customers", "wholesale_customers.zip",
                         _zip_bytes({"Wholesale customers data.csv": df.to_csv(index=False)}))
        ds = L.load_dataset("wholesale_customers", allow_download=False)
        assert "Region" not in ds.features.columns and "Channel" not in ds.features.columns
        assert ds.label_role == "external_eval_only"

    def test_air_quality_format_and_missing_marker(self):
        rows = ["Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;"]
        for h in range(48):
            co = "-200" if h == 5 else f"{2 + h % 3},{h % 10}"
            rows.append(f"10/03/2004;{h % 24:02d}.00.00;{co};1360;150;11,9;1046;166;1056;113;1692;1268;13,6;48,9;0,7578;;")
        # make dates increase
        rows = [rows[0]] + [r.replace("10/03/2004", "11/03/2004") if i >= 24 else r for i, r in enumerate(rows[1:])]
        _install_fixture("air_quality", "air_quality.zip", _zip_bytes({"AirQualityUCI.csv": "\n".join(rows)}))
        ds = L.load_dataset("air_quality", allow_download=False)
        assert "NMHC(GT)" not in ds.features.columns
        assert ds.features["CO(GT)"].isna().sum() == 1  # -200 -> NaN
        assert ds.features["T"].iloc[0] == pytest.approx(13.6)  # decimal comma
        assert pd.Index(ds.time_index).is_monotonic_increasing

    def test_malformed_source_raises_clear_error(self):
        _install_fixture("wholesale_customers", "wholesale_customers.zip",
                         _zip_bytes({"Wholesale customers data.csv": "a,b\n1,2\n"}))
        with pytest.raises(L.DatasetLoadError, match="Unexpected Wholesale columns"):
            L.load_dataset("wholesale_customers", allow_download=False)

    def test_nab_label_windows(self):
        ts = pd.date_range("2014-01-01", periods=200, freq="5min")
        csv = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": np.arange(200.0)}).to_csv(index=False)
        _install_fixture("nab_machine_temperature", "machine_temperature_system_failure.csv", csv.encode())
        win = {L._NAB_KEY: [[str(ts[50]) + ".000000", str(ts[59]) + ".000000"]]}
        (L.raw_dir("nab_machine_temperature") / "combined_windows.json").write_text(json.dumps(win))
        ds = L.load_dataset("nab_machine_temperature", allow_download=False)
        assert ds.target.sum() == 10 and ds.target.iloc[50] == 1 and ds.target.iloc[49] == 0


# ----------------------------------------------------------------- schema validation
class TestSchemaValidation:
    def test_detects_missing_column_and_bad_target(self):
        from src.data.validation import validate_loaded_dataset

        ds = L.load_dataset("breast_cancer", use_memory_cache=False, validate=False)
        ds.features = ds.features.drop(columns=[ds.features.columns[0]])
        ds.spec = get_dataset_spec("iris")  # wrong schema on purpose
        problems = validate_loaded_dataset(ds)
        assert any("missing expected columns" in p for p in problems)

    def test_detects_infinite_values(self):
        from src.data.validation import validate_loaded_dataset

        ds = L.load_dataset("iris", use_memory_cache=False)
        ds.features.iloc[0, 0] = np.inf
        assert any("infinite" in p for p in validate_loaded_dataset(ds))


# ----------------------------------------------------------------- uploads
def _csv(df):
    return df.to_csv(index=False).encode()


class TestUploadValidation:
    def test_accepts_valid_classification_csv(self):
        df = pd.DataFrame({"a": range(50), "b": np.linspace(0, 1, 50), "label": ["x", "y"] * 25})
        out, rep = validate_uploaded_csv(_csv(df), "classification", "label")
        assert rep["rows"] == 50 and out.shape == (50, 3)

    @pytest.mark.parametrize("raw,task,target,msg", [
        (b"", "regression", "y", "Empty"),
        (b"\x00\x01binary", "regression", "y", "binary"),
        (b"a,y\n1,2\n", "regression", "y", "at least"),
    ])
    def test_rejects_bad_files(self, raw, task, target, msg):
        with pytest.raises(UploadValidationError, match=msg):
            validate_uploaded_csv(raw, task, target)

    def test_rejects_missing_target(self):
        df = pd.DataFrame({"a": range(30)})
        with pytest.raises(UploadValidationError, match="not found"):
            validate_uploaded_csv(_csv(df), "regression", "price")

    def test_rejects_non_numeric_regression_target(self):
        df = pd.DataFrame({"a": range(30), "y": ["v"] * 30})
        with pytest.raises(UploadValidationError, match="numeric"):
            validate_uploaded_csv(_csv(df), "regression", "y")

    def test_rejects_unsorted_time_series(self):
        df = pd.DataFrame({"t": pd.date_range("2020", periods=30, freq="h")[::-1].astype(str), "y": range(30)})
        with pytest.raises(UploadValidationError, match="sorted"):
            validate_uploaded_csv(_csv(df), "forecasting", "y", "t")

    def test_sanitises_formula_like_column_names(self):
        df = pd.DataFrame({"=cmd()": range(30), "y": range(30)})
        out, rep = validate_uploaded_csv(_csv(df), "regression", "y")
        assert not any(str(c).startswith("=") for c in out.columns)
        assert rep["renamed_columns"]

    def test_upload_roundtrip_through_loader(self):
        from src.data.uploads import save_upload

        df = pd.DataFrame({"x1": np.arange(60.0), "x2": np.arange(60.0) ** 0.5, "y": np.arange(60.0) * 2})
        meta = save_upload(_csv(df), "my.csv", "regression", "y")
        ds = L.load_dataset(meta["id"])
        assert ds.n_samples == 60 and "y" not in ds.features.columns
        assert L.dataset_status(meta["id"])["state"] == "available"
