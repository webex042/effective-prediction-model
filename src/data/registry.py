"""
EPM Dataset Registry
====================

A single, centralised catalogue of every dataset EPM can train/evaluate on.

Each entry is a :class:`DatasetSpec` holding the metadata the platform needs to
(a) show the dataset to a user, (b) decide which algorithms may use it, and
(c) load + preprocess it reproducibly.

INTEGRITY RULES
---------------
* Every public dataset lists the *official* source page and the exact download
  mechanism (bundled with scikit-learn, scikit-learn's official fetcher, the
  OpenML API by numeric ``data_id``, or the UCI "static/public" archive URL).
  All URLs below were checked against the source pages before being added.
* A dataset is never silently substituted. If a source fails the loader raises
  :class:`~src.data.loaders.DatasetLoadError` with the reason.
* Ground-truth labels of clustering / anomaly datasets are marked
  ``label_role="external_eval_only"`` / ``"anomaly_label"`` so the benchmark
  runner never feeds them to an unsupervised learner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Task identifiers used throughout the benchmarking package.
TASKS: Tuple[str, ...] = (
    "regression",
    "classification",
    "clustering",
    "forecasting",
    "anomaly_detection",
)

SIZE_CLASSES = ("small", "medium", "large")


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one dataset."""

    id: str
    name: str
    description: str
    tasks: Tuple[str, ...]
    source: str
    source_url: str
    download_url: Optional[str]
    download_method: str  # bundled | sklearn_fetcher | openml | uci_zip | github_raw | synthetic
    license: str
    usage_notes: str
    numeric_features: Tuple[str, ...] = ()
    categorical_features: Tuple[str, ...] = ()
    target: Optional[str] = None
    target_type: Optional[str] = None  # continuous | binary | multiclass | series | anomaly_flag
    label_role: str = "target"  # target | external_eval_only | anomaly_label
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    missing_values: str = "none"
    approx_download_mb: float = 0.0
    approx_memory_mb: float = 1.0
    size_class: str = "small"
    preprocessing: Tuple[str, ...] = ()
    metrics: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    split_strategy: str = "stratified"  # stratified | random | chronological
    requires_network: bool = False
    optional_dependency: Optional[str] = None
    citation: str = ""
    caveats: Tuple[str, ...] = ()
    # Time-series specific
    time_column: Optional[str] = None
    frequency: Optional[str] = None
    seasonal_period: Optional[int] = None
    default_window: Optional[int] = None
    default_horizon: Optional[int] = None
    # Clustering / anomaly specific
    n_clusters_hint: Optional[int] = None
    contamination_prior: Optional[float] = None
    # Fairness
    sensitive_features: Tuple[str, ...] = ()
    dropped_columns: Tuple[str, ...] = ()

    def supports(self, task: str) -> bool:
        return task in self.tasks

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        d["metrics"] = {k: list(v) for k, v in self.metrics.items()}
        return d


# -----------------------------------------------------------------------------
# Shared metric / preprocessing descriptions
# -----------------------------------------------------------------------------
REG_METRICS = ("mae", "mse", "rmse", "r2")
CLF_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")
CLF_IMB_METRICS = CLF_METRICS + ("pr_auc",)
CLU_METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz", "ari_external", "nmi_external")
FC_METRICS = ("mae", "rmse", "smape", "mase", "skill_vs_naive")
AD_METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc")
AD_SERIES_METRICS = AD_METRICS + ("detection_delay",)

TAB_PREP = (
    "drop exact duplicate rows (before splitting)",
    "median imputation for numeric columns (fit on train only)",
    "standard scaling for scale-sensitive models only (fit on train only)",
)
CAT_PREP = TAB_PREP + (
    "most-frequent imputation + one-hot encoding with handle_unknown='ignore' (fit on train only)",
)
TS_PREP = (
    "chronological train/val/test split (no shuffling)",
    "causal forward-fill of gaps (past values only)",
    "per-channel standardisation fit on the training segment only",
    "sliding windows (lookback W -> horizon H) built inside each segment",
)

UCI_CC_BY = "CC BY 4.0 (UCI Machine Learning Repository)"


def _spec(**kw: Any) -> DatasetSpec:
    return DatasetSpec(**kw)


_SPECS: List[DatasetSpec] = [
    # ======================= REGRESSION =====================================
    _spec(
        id="california_housing",
        name="California Housing",
        description="1990 U.S. census block groups in California; predict median house value (in $100k) from income, age and occupancy features.",
        tasks=("regression",),
        source="scikit-learn (StatLib repository)",
        source_url="https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset",
        download_url=None,
        download_method="sklearn_fetcher",
        license="Public (StatLib); distributed via scikit-learn's official fetcher",
        usage_notes="Downloaded once by sklearn.datasets.fetch_california_housing and cached under data/raw/sklearn.",
        numeric_features=("MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"),
        target="MedHouseVal", target_type="continuous",
        n_samples=20640, n_features=8, missing_values="none",
        approx_download_mb=0.4, approx_memory_mb=1.5, size_class="medium",
        preprocessing=TAB_PREP, metrics={"regression": REG_METRICS},
        split_strategy="random", requires_network=True,
        citation="Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297.",
        caveats=("Target is capped at 5.0 ($500k) which biases errors at the top end.",),
    ),
    _spec(
        id="diabetes",
        name="Diabetes (Efron et al.)",
        description="442 diabetes patients; predict a quantitative measure of disease progression one year after baseline from 10 baseline variables.",
        tasks=("regression",),
        source="scikit-learn (bundled)",
        source_url="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
        download_url=None, download_method="bundled",
        license="BSD-3-Clause (bundled with scikit-learn)",
        usage_notes="Features are already mean-centred and scaled by scikit-learn.",
        numeric_features=("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"),
        target="progression", target_type="continuous",
        n_samples=442, n_features=10, approx_memory_mb=0.1, size_class="small",
        preprocessing=TAB_PREP, metrics={"regression": REG_METRICS}, split_strategy="random",
        citation="Efron, Hastie, Johnstone and Tibshirani (2004) Least Angle Regression, Annals of Statistics.",
        caveats=("Small sample size: prefer cross-validated scores over a single test split.",),
    ),
    _spec(
        id="energy_efficiency",
        name="UCI Energy Efficiency",
        description="768 simulated building shapes; predict heating load (Y1) from 8 building design parameters (cooling load Y2 is kept out of the features).",
        tasks=("regression",),
        source="UCI Machine Learning Repository (id 242)",
        source_url="https://archive.ics.uci.edu/dataset/242/energy+efficiency",
        download_url="https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
        download_method="uci_zip", license=UCI_CC_BY,
        usage_notes="Archive contains ENB2012_data.xlsx; reading it requires the optional 'openpyxl' package.",
        numeric_features=("relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height", "orientation", "glazing_area", "glazing_area_distribution"),
        target="heating_load", target_type="continuous",
        n_samples=768, n_features=8, approx_download_mb=0.08, approx_memory_mb=0.1, size_class="small",
        preprocessing=TAB_PREP, metrics={"regression": REG_METRICS}, split_strategy="random",
        requires_network=True, optional_dependency="openpyxl",
        citation="A. Tsanas, A. Xifara (2012) Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools, Energy and Buildings 49:560-567.",
        dropped_columns=("cooling_load",),
    ),
    _spec(
        id="epm_sensor_regression",
        name="EPM Synthetic Sensor Regression",
        description="EPM's existing Stage 2 synthetic machine-sensor dataset: predict core temperature from vibration, pressure, RPM, load and ambient temperature.",
        tasks=("regression",),
        source="Synthetic (src/classical_ml/baselines.py)",
        source_url="src/classical_ml/baselines.py", download_url=None, download_method="synthetic",
        license="Project-generated (MIT, same as EPM)",
        usage_notes="Kept for continuity with the Stage 2 leaderboard. Known generating function, so useful for sanity checks only.",
        numeric_features=("Vibration", "Pressure", "RPM", "Load", "Ambient_Temp"),
        target="Core_Temp", target_type="continuous",
        n_samples=1200, n_features=5, approx_memory_mb=0.05, size_class="small",
        preprocessing=TAB_PREP, metrics={"regression": REG_METRICS}, split_strategy="random",
        caveats=("Synthetic: results do not transfer to real-world data.",),
    ),
    # ======================= CLASSIFICATION =================================
    _spec(
        id="iris",
        name="Iris",
        description="Fisher's 150 iris flowers, 4 measurements, 3 species. Multiclass classification; labels are used only for external evaluation when clustering.",
        tasks=("classification", "clustering"),
        source="scikit-learn (bundled; originally UCI id 53)",
        source_url="https://archive.ics.uci.edu/dataset/53/iris",
        download_url=None, download_method="bundled",
        license="CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause",
        usage_notes="Very small: use cross-validation; single-split metrics have high variance.",
        numeric_features=("sepal_length", "sepal_width", "petal_length", "petal_width"),
        target="species", target_type="multiclass",
        n_samples=150, n_features=4, approx_memory_mb=0.01, size_class="small",
        preprocessing=TAB_PREP, metrics={"classification": CLF_METRICS, "clustering": CLU_METRICS},
        split_strategy="stratified", n_clusters_hint=3,
        citation="R. A. Fisher (1936) The use of multiple measurements in taxonomic problems.",
    ),
    _spec(
        id="breast_cancer",
        name="Breast Cancer Wisconsin (Diagnostic)",
        description="569 fine-needle aspirate images described by 30 nucleus features; binary benign/malignant diagnosis.",
        tasks=("classification",),
        source="scikit-learn (bundled; originally UCI id 17)",
        source_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        download_url=None, download_method="bundled",
        license="CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause",
        usage_notes="Positive class is 'malignant' (sklearn encodes malignant=0; EPM re-encodes malignant=1 so recall means cancer detection).",
        target="diagnosis_malignant", target_type="binary",
        n_samples=569, n_features=30, approx_memory_mb=0.15, size_class="small",
        preprocessing=TAB_PREP, metrics={"classification": CLF_IMB_METRICS}, split_strategy="stratified",
        citation="W.N. Street, W.H. Wolberg and O.L. Mangasarian (1993) Nuclear feature extraction for breast tumor diagnosis.",
        caveats=("Educational benchmark only; not for clinical decision-making.",),
    ),
    _spec(
        id="wine",
        name="Wine Recognition",
        description="178 wines from three cultivars, 13 chemical-analysis features. Multiclass classification; also usable for clustering with labels held out.",
        tasks=("classification", "clustering"),
        source="scikit-learn (bundled; originally UCI id 109)",
        source_url="https://archive.ics.uci.edu/dataset/109/wine",
        download_url=None, download_method="bundled",
        license="CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause",
        usage_notes="Features have very different scales: scale-sensitive models need standardisation.",
        target="cultivar", target_type="multiclass",
        n_samples=178, n_features=13, approx_memory_mb=0.02, size_class="small",
        preprocessing=TAB_PREP, metrics={"classification": CLF_METRICS, "clustering": CLU_METRICS},
        split_strategy="stratified", n_clusters_hint=3,
        citation="Forina, M. et al. PARVUS - An Extendible Package for Data Exploration, Classification and Correlation.",
    ),
    _spec(
        id="digits",
        name="Optical Handwritten Digits (8x8)",
        description="1,797 8x8 grey-scale images of handwritten digits flattened to 64 pixel features; 10-class classification. Suits EPM's MLP (EPM has no CNN).",
        tasks=("classification",),
        source="scikit-learn (bundled; test split of UCI id 80)",
        source_url="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
        download_url=None, download_method="bundled",
        license="CC BY 4.0 (UCI); bundled with scikit-learn under BSD-3-Clause",
        usage_notes="Pixel intensities 0-16; standardisation applied for scale-sensitive models.",
        target="digit", target_type="multiclass",
        n_samples=1797, n_features=64, approx_memory_mb=0.9, size_class="small",
        preprocessing=TAB_PREP, metrics={"classification": CLF_METRICS}, split_strategy="stratified",
        citation="E. Alpaydin, C. Kaynak (1998) Optical Recognition of Handwritten Digits, UCI.",
    ),
    _spec(
        id="adult",
        name="Adult Census Income",
        description="48,842 records from the 1994 U.S. Census; predict whether income exceeds $50K/yr. Mixed numeric + categorical features with missing values.",
        tasks=("classification",),
        source="OpenML (data_id=1590, version 2; originally UCI id 2)",
        source_url="https://www.openml.org/d/1590",
        download_url="https://www.openml.org/api/v1/json/data/1590",
        download_method="openml",
        license="OpenML: Public; UCI original: CC BY 4.0",
        usage_notes="Downloaded through sklearn.datasets.fetch_openml(data_id=1590) and cached under data/raw/sklearn. 'fnlwgt' is a census sampling weight, not a personal attribute, and is dropped.",
        numeric_features=("age", "education-num", "capital-gain", "capital-loss", "hours-per-week"),
        categorical_features=("workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"),
        target="income_gt_50k", target_type="binary",
        n_samples=48842, n_features=13, missing_values="yes - workclass, occupation, native-country (NaN)",
        approx_download_mb=2.5, approx_memory_mb=25, size_class="medium",
        preprocessing=CAT_PREP, metrics={"classification": CLF_IMB_METRICS}, split_strategy="stratified",
        requires_network=True,
        citation="R. Kohavi (1996) Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid, KDD.",
        caveats=(
            "Contains protected attributes (race, sex). EPM reports per-group selection-rate and TPR gaps for these columns.",
            "1994 data with a ~24% positive rate; models encode historical income disparities and must not be used for real decisions.",
        ),
        sensitive_features=("sex", "race"),
        dropped_columns=("fnlwgt",),
    ),
    _spec(
        id="bank_marketing",
        name="UCI Bank Marketing",
        description="45,211 phone-campaign contacts of a Portuguese bank; predict whether the client subscribed a term deposit (~11.7% positive).",
        tasks=("classification",),
        source="OpenML (data_id=1461; originally UCI id 222, bank-full)",
        source_url="https://www.openml.org/d/1461",
        download_url="https://www.openml.org/api/v1/json/data/1461",
        download_method="openml",
        license="OpenML: Public; UCI original: CC BY 4.0",
        usage_notes="OpenML names columns V1..V16 and Class (1=no, 2=yes); EPM restores the UCI names. 'duration' is dropped because the UCI documentation states it is only known after the call and leaks the outcome.",
        numeric_features=("age", "balance", "day", "campaign", "pdays", "previous"),
        categorical_features=("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"),
        target="subscribed", target_type="binary",
        n_samples=45211, n_features=15, missing_values="none (unknowns encoded as the category 'unknown')",
        approx_download_mb=1.5, approx_memory_mb=20, size_class="medium",
        preprocessing=CAT_PREP + ("optional class_weight='balanced' for imbalance",),
        metrics={"classification": CLF_IMB_METRICS}, split_strategy="stratified", requires_network=True,
        citation="S. Moro, P. Cortez and P. Rita (2014) A Data-Driven Approach to Predict the Success of Bank Telemarketing, Decision Support Systems.",
        caveats=("Imbalanced: accuracy is misleading; compare PR-AUC / F1 / balanced accuracy.",),
        dropped_columns=("duration",),
    ),
    _spec(
        id="epm_sensor_failure",
        name="EPM Synthetic Equipment Failure",
        description="EPM's existing Stage 2 synthetic binary failure dataset (25% failures) from 5 sensor channels.",
        tasks=("classification",),
        source="Synthetic (src/classical_ml/baselines.py)",
        source_url="src/classical_ml/baselines.py", download_url=None, download_method="synthetic",
        license="Project-generated (MIT, same as EPM)",
        usage_notes="Kept for continuity with the Stage 2 leaderboard.",
        numeric_features=("Vibration", "Pressure", "RPM", "Load", "Ambient_Temp"),
        target="failure", target_type="binary",
        n_samples=1200, n_features=5, approx_memory_mb=0.05, size_class="small",
        preprocessing=TAB_PREP, metrics={"classification": CLF_IMB_METRICS}, split_strategy="stratified",
        caveats=("Synthetic: results do not transfer to real-world data.",),
    ),
    # ======================= CLUSTERING =====================================
    _spec(
        id="wholesale_customers",
        name="UCI Wholesale Customers",
        description="440 clients of a Portuguese wholesale distributor; annual spending (m.u.) on 6 product categories. Channel (HoReCa/Retail) kept only for external evaluation.",
        tasks=("clustering",),
        source="UCI Machine Learning Repository (id 292)",
        source_url="https://archive.ics.uci.edu/dataset/292/wholesale+customers",
        download_url="https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip",
        download_method="uci_zip", license=UCI_CC_BY,
        usage_notes="Spending is heavily right-skewed; a log1p transform is applied before scaling.",
        numeric_features=("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"),
        target="Channel", target_type="multiclass", label_role="external_eval_only",
        n_samples=440, n_features=6, approx_download_mb=0.01, approx_memory_mb=0.03, size_class="small",
        preprocessing=("log1p transform of spending", "standard scaling"),
        metrics={"clustering": CLU_METRICS}, split_strategy="random", requires_network=True,
        n_clusters_hint=2,
        citation="M. G. M. S. Cardoso (2014) Wholesale customers, UCI Machine Learning Repository.",
    ),
    _spec(
        id="seeds",
        name="UCI Seeds",
        description="210 wheat kernels from three varieties (Kama, Rosa, Canadian); 7 geometric features. Variety is held out for external evaluation when clustering.",
        tasks=("clustering", "classification"),
        source="UCI Machine Learning Repository (id 236)",
        source_url="https://archive.ics.uci.edu/dataset/236/seeds",
        download_url="https://archive.ics.uci.edu/static/public/236/seeds.zip",
        download_method="uci_zip", license=UCI_CC_BY,
        usage_notes="seeds_dataset.txt is whitespace-separated with occasional repeated tabs; parsed with a regex separator.",
        numeric_features=("area", "perimeter", "compactness", "kernel_length", "kernel_width", "asymmetry", "groove_length"),
        target="variety", target_type="multiclass",
        n_samples=210, n_features=7, approx_download_mb=0.01, approx_memory_mb=0.02, size_class="small",
        preprocessing=TAB_PREP, metrics={"clustering": CLU_METRICS, "classification": CLF_METRICS},
        split_strategy="stratified", requires_network=True, n_clusters_hint=3,
        citation="M. Charytanowicz et al. (2010) Complete Gradient Clustering Algorithm for Features Analysis of X-ray Images.",
    ),
    _spec(
        id="synthetic_blobs",
        name="Synthetic Gaussian Blobs",
        description="Controlled clustering benchmark: 1,500 points from 4 isotropic Gaussian clusters in 6-D (sklearn.datasets.make_blobs, fixed seed).",
        tasks=("clustering",),
        source="Synthetic (sklearn.datasets.make_blobs)",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html",
        download_url=None, download_method="synthetic", license="Project-generated",
        usage_notes="Known ground truth (4 clusters) for sanity-checking clustering metrics.",
        numeric_features=tuple(f"x{i}" for i in range(6)),
        target="true_cluster", target_type="multiclass", label_role="external_eval_only",
        n_samples=1500, n_features=6, approx_memory_mb=0.1, size_class="small",
        preprocessing=("standard scaling",), metrics={"clustering": CLU_METRICS},
        split_strategy="random", n_clusters_hint=4,
    ),
    # ======================= FORECASTING ====================================
    _spec(
        id="household_power",
        name="Individual Household Electric Power Consumption",
        description="One household near Paris, Dec 2006 - Nov 2010, 1-minute readings (2,075,259 rows) resampled to hourly means; forecast Global_active_power.",
        tasks=("forecasting",),
        source="UCI Machine Learning Repository (id 235)",
        source_url="https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        download_url="https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        download_method="uci_zip", license=UCI_CC_BY,
        usage_notes="19.7 MB zip / 126.8 MB text. Parsed once, resampled to hourly means and cached as data/processed/household_power_hourly.csv.gz.",
        numeric_features=("Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"),
        target="Global_active_power", target_type="series",
        n_samples=34589, n_features=7, missing_values="~1.25% of minutes ('?' or empty); hours with no readings are forward-filled causally",
        approx_download_mb=19.7, approx_memory_mb=300, size_class="large",
        preprocessing=TS_PREP + ("resample 1-min -> hourly mean",),
        metrics={"forecasting": FC_METRICS}, split_strategy="chronological", requires_network=True,
        time_column="datetime", frequency="1h", seasonal_period=24, default_window=48, default_horizon=1,
        citation="G. Hebrail, A. Berard (2012) Individual household electric power consumption, UCI.",
        caveats=("First load parses ~2M rows (tens of seconds, ~300 MB peak); later loads use the hourly cache.",),
    ),
    _spec(
        id="air_quality",
        name="UCI Air Quality",
        description="9,358 hourly averages from a chemical multisensor device in an Italian city (Mar 2004 - Feb 2005); forecast CO(GT) concentration.",
        tasks=("forecasting",),
        source="UCI Machine Learning Repository (id 360)",
        source_url="https://archive.ics.uci.edu/dataset/360/air+quality",
        download_url="https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        download_method="uci_zip",
        license=UCI_CC_BY + " - the source page states the data may be used exclusively for research purposes (no commercial use)",
        usage_notes="Missing values are tagged -200 and converted to NaN. NMHC(GT) is ~90% missing and is dropped.",
        numeric_features=("CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"),
        target="CO(GT)", target_type="series",
        n_samples=9357, n_features=12, missing_values="yes - tagged -200 (CO(GT) ~18%)",
        approx_download_mb=1.5, approx_memory_mb=2, size_class="medium",
        preprocessing=TS_PREP + ("-200 -> NaN", "drop NMHC(GT)"),
        metrics={"forecasting": FC_METRICS}, split_strategy="chronological", requires_network=True,
        time_column="datetime", frequency="1h", seasonal_period=24, default_window=24, default_horizon=1,
        citation="S. De Vito et al. (2008) On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario, Sensors and Actuators B.",
        caveats=("Gaps in the target are forward-filled from the past, which flattens some test targets; metrics on long gaps are optimistic.",),
        dropped_columns=("NMHC(GT)",),
    ),
    _spec(
        id="epm_sensor_telemetry",
        name="EPM Synthetic Sensor Telemetry",
        description="EPM's existing Stage 5/13 multivariate telemetry (temperature, vibration, pressure, RPM, load; 10-min sampling) with a degradation phase in the final quarter.",
        tasks=("forecasting",),
        source="Synthetic (src/timeseries/dataset.py)",
        source_url="src/timeseries/dataset.py", download_url=None, download_method="synthetic",
        license="Project-generated (MIT, same as EPM)",
        usage_notes="Same generator as Stage 5 and the Stage 13 H1 study, so results are comparable with those leaderboards.",
        numeric_features=("temperature", "vibration", "pressure", "rpm", "load"),
        target="temperature", target_type="series",
        n_samples=2000, n_features=5, approx_memory_mb=0.1, size_class="small",
        preprocessing=TS_PREP, metrics={"forecasting": FC_METRICS}, split_strategy="chronological",
        time_column="step", frequency="10min", seasonal_period=144, default_window=24, default_horizon=1,
        caveats=("The test segment falls inside the synthetic degradation phase (distribution shift by design).",),
    ),
    # ======================= ANOMALY DETECTION ==============================
    _spec(
        id="nab_machine_temperature",
        name="NAB Machine Temperature System Failure",
        description="Numenta Anomaly Benchmark: 22,695 5-minute temperature readings from an internal component of a large industrial machine, with 4 labelled anomaly windows (known causes).",
        tasks=("anomaly_detection",),
        source="Numenta Anomaly Benchmark (GitHub numenta/NAB)",
        source_url="https://github.com/numenta/NAB",
        download_url="https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
        download_method="github_raw",
        license="NAB repository licence (MIT-style, Copyright Numenta Inc.)",
        usage_notes="Labels come from labels/combined_windows.json in the same repository. Points are turned into lag-window feature vectors so tabular detectors can score them.",
        numeric_features=("value",),
        target="is_anomaly_window", target_type="anomaly_flag", label_role="anomaly_label",
        n_samples=22695, n_features=1, approx_download_mb=0.7, approx_memory_mb=2, size_class="medium",
        preprocessing=("lag-window features (past values only)", "standard scaling fit on the earliest 50% (training) segment"),
        metrics={"anomaly_detection": AD_SERIES_METRICS}, split_strategy="chronological", requires_network=True,
        time_column="timestamp", frequency="5min", default_window=12, contamination_prior=0.1,
        citation="A. Lavin, S. Ahmad (2015) Evaluating Real-time Anomaly Detection Algorithms - the Numenta Anomaly Benchmark, IEEE ICMLA.",
        caveats=("NAB labels whole windows around each anomaly, so point-wise precision is pessimistic; detection delay is the more faithful metric.",),
    ),
    _spec(
        id="credit_card_fraud",
        name="Credit Card Fraud Detection (ULB)",
        description="284,807 European card transactions (Sept 2013) with 492 frauds (0.172%); 28 PCA components plus Time and Amount.",
        tasks=("anomaly_detection", "classification"),
        source="OpenML (data_id=1597; originally ULB Machine Learning Group)",
        source_url="https://www.openml.org/d/1597",
        download_url="https://www.openml.org/api/v1/json/data/1597",
        download_method="openml",
        license="OpenML: Public (original release by the ULB Machine Learning Group; check the original terms before redistribution)",
        usage_notes="Large download (~150 MB ARFF). Disabled unless 'allow_large' is set. For anomaly detection the fraud label is used only for evaluation.",
        target="is_fraud", target_type="binary", label_role="anomaly_label",
        n_samples=284807, n_features=30, approx_download_mb=150, approx_memory_mb=70, size_class="large",
        preprocessing=TAB_PREP, metrics={"anomaly_detection": AD_METRICS, "classification": CLF_IMB_METRICS},
        split_strategy="stratified", requires_network=True, contamination_prior=0.0017,
        citation="A. Dal Pozzolo et al. (2015) Calibrating Probability with Undersampling for Unbalanced Classification, IEEE SSCI.",
        caveats=("Extreme imbalance: accuracy and ROC-AUC are misleading; use PR-AUC.",),
    ),
    _spec(
        id="synthetic_anomalies",
        name="Synthetic Gaussian Anomalies",
        description="Controllable anomaly benchmark: 2,000 points from 3 Gaussian inlier clusters in 8-D plus uniformly scattered outliers at a 5% rate (fixed seed).",
        tasks=("anomaly_detection",),
        source="Synthetic (NumPy, fixed seed)",
        source_url="src/data/loaders.py", download_url=None, download_method="synthetic",
        license="Project-generated",
        usage_notes="Anomaly rate is controllable via the loader's 'contamination' argument.",
        numeric_features=tuple(f"x{i}" for i in range(8)),
        target="is_anomaly", target_type="anomaly_flag", label_role="anomaly_label",
        n_samples=2000, n_features=8, approx_memory_mb=0.15, size_class="small",
        preprocessing=("standard scaling (fit on train only)",), metrics={"anomaly_detection": AD_METRICS},
        split_strategy="stratified", contamination_prior=0.05,
    ),
]

DATASETS: Dict[str, DatasetSpec] = {s.id: s for s in _SPECS}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Return the spec for ``dataset_id`` or raise ``KeyError`` with the valid ids."""
    if dataset_id in DATASETS:
        return DATASETS[dataset_id]
    if dataset_id.startswith("upload_"):
        from src.data.uploads import upload_spec

        return upload_spec(dataset_id)
    try:
        return DATASETS[dataset_id]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Known datasets: {sorted(DATASETS)}"
        ) from None


def list_datasets(task: Optional[str] = None) -> List[DatasetSpec]:
    """All registered datasets, optionally filtered to those supporting ``task``."""
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {TASKS}")
    specs = list(DATASETS.values())
    if task:
        specs = [s for s in specs if s.supports(task)]
    order = {c: i for i, c in enumerate(SIZE_CLASSES)}
    return sorted(specs, key=lambda s: (order.get(s.size_class, 9), s.n_samples or 0))


def validate_registry() -> List[str]:
    """Static consistency checks over every spec. Returns a list of problems."""
    problems: List[str] = []
    for s in DATASETS.values():
        if not s.tasks or any(t not in TASKS for t in s.tasks):
            problems.append(f"{s.id}: invalid tasks {s.tasks}")
        if s.size_class not in SIZE_CLASSES:
            problems.append(f"{s.id}: invalid size_class {s.size_class}")
        if s.requires_network and not s.download_url and s.download_method not in ("sklearn_fetcher",):
            problems.append(f"{s.id}: requires network but has no download_url")
        if s.download_url and not s.download_url.startswith("https://"):
            problems.append(f"{s.id}: download_url must be https")
        for t in s.tasks:
            if t not in s.metrics:
                problems.append(f"{s.id}: no metrics declared for task {t}")
        if "forecasting" in s.tasks and (not s.seasonal_period or not s.default_window):
            problems.append(f"{s.id}: forecasting datasets need seasonal_period and default_window")
        if not s.license:
            problems.append(f"{s.id}: missing license")
    return problems
