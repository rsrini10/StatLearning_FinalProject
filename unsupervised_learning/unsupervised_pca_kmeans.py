#!/usr/bin/env python3
"""
Baseline pipeline: scale nutrients -> PCA -> k-means on PC scores -> plots.

Data live one directory up: ../food_nutrient_conc.csv (repo root of CSV exports).

Hyperparameters are intentionally simple for a first pass; swap in CV / grid search later.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Repo layout: StatLearning_FinalProject/food_nutrient_conc.csv (parent of this folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_K_CLUSTERS = 8
DEFAULT_N_COMPONENTS_KMEANS = 15
SCREE_PLOT_COMPONENTS = 25


def load_feature_matrix(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    names = df["Food_Name"] if "Food_Name" in df.columns else pd.Series(np.arange(len(df)))
    numeric = df.select_dtypes(include=[np.number])
    drop_cols = [c for c in numeric.columns if str(c).lower().startswith("unnamed")]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")
    return numeric, names


@dataclass
class PcaKMeansResult:
    preprocess: Pipeline
    pca: PCA
    kmeans: KMeans
    Z: np.ndarray  # full PC scores (n_samples x n_components_)
    cluster_labels: np.ndarray
    food_names: pd.Series
    silhouette: float
    n_pc_kmeans: int


def fit_pca_kmeans(
    X_raw: pd.DataFrame,
    food_names: pd.Series,
    *,
    k: int = DEFAULT_K_CLUSTERS,
    pc_kmeans: int = DEFAULT_N_COMPONENTS_KMEANS,
    seed: int = 42,
) -> PcaKMeansResult:
    """Median-impute, scale, PCA, then k-means on the first `pc_kmeans` PC scores."""
    n_samples, n_features = X_raw.shape
    preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    X = preprocess.fit_transform(X_raw)

    max_pc = min(n_samples, n_features, X.shape[1])
    n_pc_kmeans = int(np.clip(pc_kmeans, 1, max_pc))

    pca_full = PCA(random_state=seed)
    Z_full = pca_full.fit_transform(X)

    Z_kmeans = Z_full[:, :n_pc_kmeans]
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(Z_kmeans)
    sil = silhouette_score(Z_kmeans, labels)

    return PcaKMeansResult(
        preprocess=preprocess,
        pca=pca_full,
        kmeans=km,
        Z=Z_full,
        cluster_labels=labels,
        food_names=food_names,
        silhouette=sil,
        n_pc_kmeans=n_pc_kmeans,
    )


def save_plots_and_csv(result: PcaKMeansResult, out_dir: Path, scree_components: int = SCREE_PLOT_COMPONENTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    evr = result.pca.explained_variance_ratio_
    Z = result.Z
    km = result.kmeans

    n_scree = min(scree_components, len(evr))
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(np.arange(1, n_scree + 1), evr[:n_scree], color="steelblue", alpha=0.85)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_title("PCA scree plot (baseline)")
    ax1.set_xticks(np.arange(1, n_scree + 1, max(1, n_scree // 10)))
    fig1.tight_layout()
    fig1.savefig(out_dir / "pca_scree.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 7))
    scatter = ax2.scatter(
        Z[:, 0],
        Z[:, 1],
        c=result.cluster_labels,
        cmap="tab10",
        alpha=0.55,
        s=12,
        linewidths=0,
    )
    ax2.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax2.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax2.set_title(f"PCA projection (K-means K={km.n_clusters}, fitted on first {result.n_pc_kmeans} PCs)")
    plt.colorbar(scatter, ax=ax2, label="cluster")
    fig2.tight_layout()
    fig2.savefig(out_dir / "pca_pc1_pc2_kmeans.png", dpi=150)
    plt.close(fig2)

    out_tbl = pd.DataFrame(
        {
            "Food_Name": result.food_names.values,
            "PC1": Z[:, 0],
            "PC2": Z[:, 1],
            "cluster": result.cluster_labels,
        }
    )
    out_tbl.to_csv(out_dir / "pca_kmeans_assignments.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA then k-means on USDA nutrient table.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "food_nutrient_conc.csv",
        help="Path to food_nutrient_conc.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for saved figures",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K_CLUSTERS)
    parser.add_argument("--pc-kmeans", type=int, default=DEFAULT_N_COMPONENTS_KMEANS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_raw, food_names = load_feature_matrix(args.data)
    print(f"Loaded {args.data.name}: {len(X_raw)} foods, {X_raw.shape[1]} numeric features")

    result = fit_pca_kmeans(X_raw, food_names, k=args.k, pc_kmeans=args.pc_kmeans, seed=args.seed)
    evr = result.pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    print(
        f"k-means: K={args.k}, PCs used={result.n_pc_kmeans}, "
        f"inertia={result.kmeans.inertia_:.2f}, silhouette={result.silhouette:.4f}"
    )
    print(
        f"Variance explained: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, "
        f"first {result.n_pc_kmeans} PCs cumulative={cum[result.n_pc_kmeans - 1]:.3f}"
    )

    save_plots_and_csv(result, args.out_dir)
    print(f"Wrote plots and CSV under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
