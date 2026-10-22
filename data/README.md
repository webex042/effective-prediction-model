# EPM — Data Directory

## Structure

```
data/
├── raw/        ← Original, unmodified datasets
├── processed/  ← Cleaned, transformed, feature-engineered data
└── README.md   ← This file
```

## Guidelines

1. **Never modify raw data.** Raw data is immutable. All transformations
   produce new files in `processed/`.

2. **Document every dataset.** For each dataset, record:
   - Source (URL, API, synthetic)
   - Download date
   - Number of samples / features
   - Target variable(s)
   - Known issues or biases
   - License

3. **Version your preprocessing.** The code in `src/preprocessing/` should
   be able to reproduce `processed/` from `raw/` deterministically.

4. **Large files stay out of git.** The `.gitignore` excludes data files.
   Document download instructions here instead.

## Datasets

All datasets are declared in `src/data/registry.py` (source, licence, schema, task, metrics).
Downloads are cached under `data/raw/<dataset_id>/` with a `manifest.json` that holds the SHA-256, the URL and the download time.
Derived caches go in `data/processed/`, scikit-learn/OpenML caches in `data/raw/sklearn/`, and user uploads in `data/raw/uploads/`.
None of these are committed.

See `docs/15_datasets_and_benchmarking.md` for the full table, the compatibility matrix and verification status.
To download and verify everything:

```bash
python run.py --verify-datasets            # writes data/verification_report.json
python run.py --verify-datasets --allow-large
```
