# Motion Pattern Clustering and Injury-Risk Anomaly Detection

Unsupervised learning pipeline for grouping motion into interpretable strain clusters and flagging high-risk movement anomalies. The repository includes a runnable synthetic-data workflow so the clustering, anomaly detection, and visualisation stack can be demonstrated immediately.

## What This Repo Includes

- K-Means clustering for low-, medium-, and high-strain grouping
- DBSCAN for density-based structure and noise detection
- Isolation Forest for high-risk anomaly flagging
- PCA and t-SNE visualisation exports for analysis and reporting
- CLI that writes summary metrics, analysis tables, saved models, and plots

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m motion_pattern.cli --output-dir reports/demo
```

## Project Structure

- `src/motion_pattern/` clustering and anomaly detection code
- `tests/` smoke test for the full demo run
- `reports/` generated plots and summary outputs
- `data/` place real movement datasets here
- `models/` reserved for persisted pipelines or exported artifacts

## Replacing The Demo Data

The current synthetic generator creates motion segments with acceleration, jerk, flexion, arm elevation, repetition, load, and asymmetry features. To adapt this to a real dataset, provide a table with the feature columns listed in `src/motion_pattern/pipeline.py` and then reuse the same unsupervised analysis flow.

## Output Artifacts

Running the demo writes:

- `reports/demo/summary.json`
- `reports/demo/motion_analysis.csv`
- `reports/demo/motion_patterns.png`
- `reports/demo/unsupervised_models.joblib`
- `reports/demo/synthetic_motion_sample.csv`
