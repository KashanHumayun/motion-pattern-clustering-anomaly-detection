# Motion Pattern Clustering and Injury-Risk Anomaly Detection

Real-data unsupervised motion analysis pipeline built on the PAMAP2 protocol recordings. The repository windows raw IMU streams, clusters movement patterns into strain bands, detects anomalies, and exports visualisations for review.

![Real-data motion patterns](reports/results/motion_patterns.png)

## Dataset

- PAMAP2 Physical Activity Monitoring
- Raw protocol recordings from 9 subjects
- Windowed IMU features from hand, chest, and ankle sensors

## Current Results

| Component | Result |
| --- | --- |
| Windows analysed | 3,829 |
| K-Means silhouette score | 0.229 |
| Strain bands | 3 learned clusters mapped to low, medium, and high strain |
| DBSCAN | 13 dense clusters with 222 noise windows |
| Isolation Forest | 307 anomaly flags |

Top anomalies are dominated by high-intensity activities such as `rope_jumping` and `running`, which is consistent with the motion intensity captured in the raw IMU signals.

## What The Pipeline Does

- loads real PAMAP2 protocol recordings
- windows and featurises multi-sensor IMU sequences
- clusters movement windows with K-Means and DBSCAN
- flags atypical motion windows with Isolation Forest
- exports PCA/t-SNE visualisations and review notes

## Run It

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m motion_pattern.cli --output-dir reports/results --model-dir models/results
```

## Output Files

- `reports/results/summary.json`
- `reports/results/motion_analysis.csv`
- `reports/results/motion_patterns.png`
- `models/results/unsupervised_models.joblib`
- `notebooks/real_data_walkthrough.ipynb`

## Notes

- Raw downloaded PAMAP2 files are kept locally under `data/raw/` and ignored by git.
- The checked-in outputs are produced from the real dataset rather than generated samples.
