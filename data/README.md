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

## Datasets (to be added per stage)

| Stage | Dataset | Type | Source | Status |
|-------|---------|------|--------|--------|
| 2 | TBD | Tabular | — | ⏳ |
| 4 | TBD | Text | — | ⏳ |
| 5 | TBD | Time-series (sensor) | — | ⏳ |
| 6 | TBD | Multimodal | — | ⏳ |
