# Motion Pattern Clustering and Injury-Risk Anomaly Detection

Unsupervised learning pipeline for clustering human motion patterns and flagging high-risk movement anomalies from wearable or body-movement data.

## Scope

- Clustering: K-Means and DBSCAN
- Anomaly detection: Isolation Forest
- Interpretation: PCA and t-SNE visualisation
- Goal: identify low-, medium-, and high-strain movement profiles without dense labels

## Planned Workflow

1. Prepare unlabelled motion data
2. Extract movement descriptors and strain indicators
3. Cluster motion patterns
4. Detect high-risk anomalous events
5. Visualise cluster structure and outlier behaviour for reporting

## Repository Structure

- `data/` for sensor and motion datasets
- `notebooks/` for exploratory analysis
- `src/` for clustering and anomaly detection code
- `models/` for saved pipelines
- `reports/` for plots and summaries
