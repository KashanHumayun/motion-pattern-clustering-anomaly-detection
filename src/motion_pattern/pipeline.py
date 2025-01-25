from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "acceleration_rms",
    "jerk_rms",
    "trunk_flexion_deg",
    "arm_elevation_deg",
    "repetition_rate",
    "symmetry_index",
    "load_proxy",
    "duration_s",
]

STRAIN_BANDS = ["low_strain", "medium_strain", "high_strain"]


def generate_synthetic_motion_dataset(
    n_samples: int = 360,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    templates = {
        "low_strain": {
            "acceleration_rms": (0.45, 0.12),
            "jerk_rms": (0.28, 0.1),
            "trunk_flexion_deg": (18.0, 6.0),
            "arm_elevation_deg": (35.0, 10.0),
            "repetition_rate": (5.0, 1.8),
            "symmetry_index": (0.14, 0.04),
            "load_proxy": (2.5, 1.0),
            "duration_s": (8.0, 2.0),
        },
        "medium_strain": {
            "acceleration_rms": (0.82, 0.16),
            "jerk_rms": (0.55, 0.14),
            "trunk_flexion_deg": (38.0, 8.0),
            "arm_elevation_deg": (62.0, 12.0),
            "repetition_rate": (8.5, 2.0),
            "symmetry_index": (0.25, 0.06),
            "load_proxy": (5.5, 1.5),
            "duration_s": (10.5, 2.5),
        },
        "high_strain": {
            "acceleration_rms": (1.28, 0.22),
            "jerk_rms": (0.92, 0.18),
            "trunk_flexion_deg": (58.0, 10.0),
            "arm_elevation_deg": (92.0, 14.0),
            "repetition_rate": (11.5, 2.2),
            "symmetry_index": (0.38, 0.08),
            "load_proxy": (8.5, 2.0),
            "duration_s": (12.0, 2.8),
        },
    }

    bands = list(templates)
    for sample_id in range(n_samples):
        band = bands[sample_id % len(bands)]
        template = templates[band]
        sample = {
            feature: float(np.clip(rng.normal(mean, std), 0.02, None))
            for feature, (mean, std) in template.items()
        }
        if rng.random() < 0.07:
            sample["acceleration_rms"] *= rng.uniform(1.6, 2.2)
            sample["jerk_rms"] *= rng.uniform(1.8, 2.5)
            sample["trunk_flexion_deg"] += rng.uniform(12.0, 25.0)
            sample["arm_elevation_deg"] += rng.uniform(10.0, 24.0)
            sample["symmetry_index"] = min(0.95, sample["symmetry_index"] + rng.uniform(0.12, 0.25))
        sample["known_strain_band"] = band
        sample["sample_id"] = f"M{sample_id:04d}"
        rows.append(sample)

    frame = pd.DataFrame(rows)
    frame["risk_score"] = (
        0.22 * frame["acceleration_rms"]
        + 0.18 * frame["jerk_rms"]
        + 0.02 * frame["trunk_flexion_deg"]
        + 0.015 * frame["arm_elevation_deg"]
        + 0.03 * frame["repetition_rate"]
        + 0.25 * frame["symmetry_index"]
        + 0.04 * frame["load_proxy"]
    )
    return frame


def run_demo(
    output_dir: Path,
    n_samples: int = 360,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_motion_dataset(n_samples=n_samples)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(frame[FEATURE_COLUMNS])

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    frame["kmeans_cluster"] = kmeans.fit_predict(scaled_features)
    cluster_band_map = _map_clusters_to_bands(frame)
    frame["strain_band"] = frame["kmeans_cluster"].map(cluster_band_map)

    dbscan = DBSCAN(eps=1.1, min_samples=8)
    frame["dbscan_cluster"] = dbscan.fit_predict(scaled_features)

    isolation_forest = IsolationForest(contamination=0.08, random_state=42)
    frame["is_anomaly"] = isolation_forest.fit_predict(scaled_features) == -1
    frame["anomaly_score"] = -isolation_forest.score_samples(scaled_features)
    frame["review_note"] = frame.apply(_build_review_note, axis=1)

    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(scaled_features)
    frame["pca_1"] = pca_components[:, 0]
    frame["pca_2"] = pca_components[:, 1]

    perplexity = max(5, min(30, len(frame) // 12))
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=perplexity, random_state=42)
    tsne_components = tsne.fit_transform(scaled_features)
    frame["tsne_1"] = tsne_components[:, 0]
    frame["tsne_2"] = tsne_components[:, 1]

    _save_visualisations(frame, output_dir)
    frame.to_csv(output_dir / "motion_analysis.csv", index=False)
    frame.head(500).to_csv(output_dir / "synthetic_motion_sample.csv", index=False)
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "dbscan": dbscan,
            "isolation_forest": isolation_forest,
            "cluster_band_map": cluster_band_map,
        },
        output_dir / "unsupervised_models.joblib",
    )

    summary = {
        "samples": int(len(frame)),
        "kmeans": {
            "silhouette_score": float(silhouette_score(scaled_features, frame["kmeans_cluster"])),
            "cluster_to_band": {str(key): value for key, value in cluster_band_map.items()},
            "band_counts": frame["strain_band"].value_counts().to_dict(),
        },
        "dbscan": {
            "clusters_found": int(len(set(frame["dbscan_cluster"])) - (1 if -1 in set(frame["dbscan_cluster"]) else 0)),
            "noise_points": int((frame["dbscan_cluster"] == -1).sum()),
        },
        "anomaly_detection": {
            "anomaly_count": int(frame["is_anomaly"].sum()),
            "top_anomalies": frame.sort_values("anomaly_score", ascending=False)
            .head(5)[["sample_id", "strain_band", "anomaly_score", "review_note"]]
            .to_dict(orient="records"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)

    return summary


def _map_clusters_to_bands(frame: pd.DataFrame) -> dict[int, str]:
    ordered_clusters = (
        frame.groupby("kmeans_cluster")["risk_score"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )
    return {cluster: STRAIN_BANDS[index] for index, cluster in enumerate(ordered_clusters)}


def _build_review_note(row: pd.Series) -> str:
    notes: list[str] = []
    if row["is_anomaly"]:
        notes.append("Flag for manual review.")
    if row["trunk_flexion_deg"] > 60:
        notes.append("Extreme trunk flexion observed.")
    if row["arm_elevation_deg"] > 100:
        notes.append("Sustained elevated arm posture.")
    if row["jerk_rms"] > 1.2:
        notes.append("High jerk event suggests abrupt movement.")
    if row["symmetry_index"] > 0.45:
        notes.append("Asymmetric loading pattern detected.")
    if not notes:
        notes.append("Typical movement signature for cluster.")
    return " ".join(notes)


def _save_visualisations(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    color_map = {
        "low_strain": "#2a9d8f",
        "medium_strain": "#e9c46a",
        "high_strain": "#e76f51",
    }
    for band, subset in frame.groupby("strain_band", sort=False):
        axes[0].scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=28,
            alpha=0.75,
            label=band,
            color=color_map.get(str(band), "#457b9d"),
        )
    axes[0].set_title("PCA View of Motion Clusters")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    anomaly_colors = np.where(frame["is_anomaly"], "#d62828", "#577590")
    axes[1].scatter(frame["tsne_1"], frame["tsne_2"], s=26, alpha=0.75, c=anomaly_colors)
    axes[1].set_title("t-SNE View With Anomaly Flags")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    fig.tight_layout()
    fig.savefig(output_dir / "motion_patterns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
