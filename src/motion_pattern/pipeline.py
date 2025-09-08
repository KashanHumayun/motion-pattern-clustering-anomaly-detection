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
from sklearn.preprocessing import StandardScaler

PAMAP2_ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

PAMAP2_COLUMNS = ["timestamp", "activity_id", "heart_rate"]
for _sensor in ["hand", "chest", "ankle"]:
    PAMAP2_COLUMNS.extend(
        [
            f"{_sensor}_temp",
            f"{_sensor}_acc16_x",
            f"{_sensor}_acc16_y",
            f"{_sensor}_acc16_z",
            f"{_sensor}_acc6_x",
            f"{_sensor}_acc6_y",
            f"{_sensor}_acc6_z",
            f"{_sensor}_gyro_x",
            f"{_sensor}_gyro_y",
            f"{_sensor}_gyro_z",
            f"{_sensor}_mag_x",
            f"{_sensor}_mag_y",
            f"{_sensor}_mag_z",
            f"{_sensor}_ori_1",
            f"{_sensor}_ori_2",
            f"{_sensor}_ori_3",
            f"{_sensor}_ori_4",
        ]
    )

SENSOR_COLUMNS = [
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
]

STRAIN_BANDS = ["low_strain", "medium_strain", "high_strain"]


def find_dataset_root(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "pamap2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol"


def extract_window_features(window: np.ndarray) -> np.ndarray:
    features: list[float] = []
    for channel in range(window.shape[1]):
        signal = window[:, channel]
        features.extend(
            [
                float(signal.mean()),
                float(signal.std()),
                float(np.sqrt(np.mean(signal**2))),
            ]
        )
    return np.asarray(features, dtype=float)


def load_windowed_motion_data(
    protocol_root: Path,
    subject_limit: int | None = None,
    downsample: int = 5,
    window_size: int = 200,
    step_size: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subject_paths = sorted(protocol_root.glob("subject10*.dat"))
    if subject_limit is not None:
        subject_paths = subject_paths[:subject_limit]

    use_columns = ["activity_id", *SENSOR_COLUMNS]
    for subject_path in subject_paths:
        frame = pd.read_csv(
            subject_path,
            sep=r"\s+",
            header=None,
            names=PAMAP2_COLUMNS,
            usecols=use_columns,
            na_values="NaN",
        )
        frame = frame[frame["activity_id"].isin(PAMAP2_ACTIVITY_NAMES)].dropna().iloc[::downsample].reset_index(drop=True)
        values = frame[SENSOR_COLUMNS].to_numpy(dtype=float)
        labels = frame["activity_id"].map(PAMAP2_ACTIVITY_NAMES).to_numpy(dtype=object)

        for start in range(0, len(frame) - window_size + 1, step_size):
            stop = start + window_size
            window = values[start:stop]
            activity = pd.Series(labels[start:stop]).mode().iat[0]
            feature_vector = extract_window_features(window)
            row = {
                "subject_id": subject_path.stem,
                "window_start": int(start),
                "window_stop": int(stop),
                "activity_name": str(activity),
            }
            for index, value in enumerate(feature_vector):
                row[f"f_{index:02d}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_risk_score(feature_frame: pd.DataFrame) -> pd.Series:
    feature_values = feature_frame.filter(like="f_")
    rms_like = feature_values.iloc[:, 2::3].mean(axis=1)
    std_like = feature_values.iloc[:, 1::3].mean(axis=1)
    return 0.6 * rms_like + 0.4 * std_like


def build_review_note(row: pd.Series) -> str:
    notes: list[str] = []
    if row["is_anomaly"]:
        notes.append("Flag for manual review.")
    if row["strain_band"] == "high_strain":
        notes.append("Assigned to the highest motion-strain cluster.")
    if row["activity_name"] in {"rope_jumping", "running", "playing_soccer", "ascending_stairs", "descending_stairs"}:
        notes.append("Contains high-intensity locomotion.")
    if row["activity_name"] in {"vacuum_cleaning", "house_cleaning", "folding_laundry", "ironing"}:
        notes.append("Household task pattern may include repetitive upper-body motion.")
    if not notes:
        notes.append("Typical motion signature for its cluster.")
    return " ".join(notes)


def map_clusters_to_bands(frame: pd.DataFrame) -> dict[int, str]:
    ordered = frame.groupby("kmeans_cluster")["risk_score"].mean().sort_values().index.tolist()
    return {cluster: STRAIN_BANDS[index] for index, cluster in enumerate(ordered)}


def save_visualisation(frame: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "low_strain": "#2a9d8f",
        "medium_strain": "#e9c46a",
        "high_strain": "#e76f51",
    }
    for band, subset in frame.groupby("strain_band", sort=False):
        axes[0].scatter(subset["pca_1"], subset["pca_2"], s=18, alpha=0.7, label=band, color=colors.get(band, "#577590"))
    axes[0].set_title("PCA View of PAMAP2 Motion Windows")
    axes[0].set_xlabel("PCA 1")
    axes[0].set_ylabel("PCA 2")
    axes[0].legend()

    anomaly_colors = np.where(frame["is_anomaly"], "#d62828", "#577590")
    axes[1].scatter(frame["tsne_1"], frame["tsne_2"], s=18, alpha=0.7, c=anomaly_colors)
    axes[1].set_title("t-SNE View With Anomaly Flags")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    project_root: Path,
    output_dir: Path,
    model_dir: Path,
    subject_limit: int | None = None,
) -> dict[str, Any]:
    dataset_root = find_dataset_root(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = load_windowed_motion_data(dataset_root, subject_limit=subject_limit)
    feature_columns = [column for column in frame.columns if column.startswith("f_")]
    feature_matrix = frame[feature_columns].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    frame["kmeans_cluster"] = kmeans.fit_predict(scaled)
    frame["risk_score"] = compute_risk_score(frame)
    cluster_map = map_clusters_to_bands(frame)
    frame["strain_band"] = frame["kmeans_cluster"].map(cluster_map)

    dbscan = DBSCAN(eps=3.5, min_samples=10)
    frame["dbscan_cluster"] = dbscan.fit_predict(scaled)

    isolation_forest = IsolationForest(contamination=0.08, random_state=42)
    frame["is_anomaly"] = isolation_forest.fit_predict(scaled) == -1
    frame["anomaly_score"] = -isolation_forest.score_samples(scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(scaled)
    frame["pca_1"] = pca_components[:, 0]
    frame["pca_2"] = pca_components[:, 1]

    tsne_sample = min(len(frame), 1500)
    tsne_indices = np.linspace(0, len(frame) - 1, tsne_sample, dtype=int)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, tsne_sample // 20)), random_state=42)
    tsne_components = tsne.fit_transform(scaled[tsne_indices])
    frame["tsne_1"] = np.nan
    frame["tsne_2"] = np.nan
    frame.loc[tsne_indices, "tsne_1"] = tsne_components[:, 0]
    frame.loc[tsne_indices, "tsne_2"] = tsne_components[:, 1]
    frame["tsne_1"] = frame["tsne_1"].interpolate(limit_direction="both")
    frame["tsne_2"] = frame["tsne_2"].interpolate(limit_direction="both")

    frame["review_note"] = frame.apply(build_review_note, axis=1)
    frame.to_csv(output_dir / "motion_analysis.csv", index=False)
    save_visualisation(frame, output_dir / "motion_patterns.png")

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "dbscan": dbscan,
            "isolation_forest": isolation_forest,
            "cluster_map": cluster_map,
            "feature_columns": feature_columns,
        },
        model_dir / "unsupervised_models.joblib",
    )

    summary = {
        "windows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "kmeans": {
            "silhouette_score": float(silhouette_score(scaled, frame["kmeans_cluster"])),
            "cluster_to_band": {str(key): value for key, value in cluster_map.items()},
            "band_counts": frame["strain_band"].value_counts().to_dict(),
        },
        "dbscan": {
            "clusters_found": int(len(set(frame["dbscan_cluster"])) - (1 if -1 in set(frame["dbscan_cluster"]) else 0)),
            "noise_points": int((frame["dbscan_cluster"] == -1).sum()),
        },
        "anomaly_detection": {
            "anomaly_count": int(frame["is_anomaly"].sum()),
            "top_anomalies": frame.sort_values("anomaly_score", ascending=False)
            .head(5)[["subject_id", "activity_name", "strain_band", "anomaly_score", "review_note"]]
            .to_dict(orient="records"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)
    return summary


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
