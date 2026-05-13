#!/usr/bin/env python3
"""
Baseline pipeline: scale nutrients -> PCA -> k-means on PC scores -> plots.

Data live one directory up: ../food_nutrient_conc.csv (repo root of CSV exports).

By default, only **micronutrient** columns are used (same rules as R/nutrient_definitions.R);
use --all-numeric-nutrients for every numeric column in the CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nutrient_definitions import micronutrient_column_names

# Repo layout: StatLearning_FinalProject/food_nutrient_conc.csv (parent of this folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR_UNSUPERVISED = PROJECT_ROOT / "plots" / "unsupervised"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_K_CLUSTERS = 8
DEFAULT_N_COMPONENTS_KMEANS = 15
SCREE_PLOT_COMPONENTS = 25


def load_feature_matrix(
    csv_path: Path,
    *,
    micronutrients_only: bool = True,
    y_col: str | None = "Energy",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load numeric nutrient columns. By default keep only micronutrients (same rules as
    R/nutrient_definitions.R): drop macros, fatty-acid detail columns, Food_Name,
    and ``Energy`` if ``y_col="Energy"``. Set ``micronutrients_only=False`` for all
    numeric columns (legacy behavior).
    """
    df = pd.read_csv(csv_path)
    names = df["Food_Name"] if "Food_Name" in df.columns else pd.Series(np.arange(len(df)))
    numeric = df.select_dtypes(include=[np.number])
    drop_cols = [c for c in numeric.columns if str(c).lower().startswith("unnamed")]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")
    if micronutrients_only:
        keep = micronutrient_column_names(df.columns.astype(str).tolist(), y_col=y_col)
        keep_num = [c for c in keep if c in numeric.columns]
        numeric = numeric[keep_num]
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


def fit_preprocess_pca(
    X_raw: pd.DataFrame,
    food_names: pd.Series,
    *,
    pc_kmeans: int = DEFAULT_N_COMPONENTS_KMEANS,
    seed: int = 42,
) -> tuple[Pipeline, PCA, np.ndarray, int]:
    """Median-impute, scale, full PCA; return PC score matrix and ``n_pc_kmeans`` cap."""
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
    return preprocess, pca_full, Z_full, n_pc_kmeans


def kmeans_on_pc_block(
    Z_full: np.ndarray,
    n_pc_kmeans: int,
    k: int,
    seed: int,
) -> tuple[KMeans, np.ndarray, float]:
    """k-means on ``Z_full[:, :n_pc_kmeans]``."""
    Z_k = Z_full[:, :n_pc_kmeans]
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(Z_k)
    sil = silhouette_score(Z_k, labels)
    return km, labels, sil


def fit_pca_kmeans(
    X_raw: pd.DataFrame,
    food_names: pd.Series,
    *,
    k: int = DEFAULT_K_CLUSTERS,
    pc_kmeans: int = DEFAULT_N_COMPONENTS_KMEANS,
    seed: int = 42,
) -> PcaKMeansResult:
    """Median-impute, scale, PCA, then k-means on the first `pc_kmeans` PC scores."""
    preprocess, pca_full, Z_full, n_pc_kmeans = fit_preprocess_pca(
        X_raw, food_names, pc_kmeans=pc_kmeans, seed=seed
    )
    km, labels, sil = kmeans_on_pc_block(Z_full, n_pc_kmeans, k, seed)

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


def _discrete_cmap(n_cluster: int):
    """Colormap that supports K > 10 (``tab10`` is too small)."""
    n = max(int(n_cluster), 2)
    try:
        base = mpl.colormaps["turbo"]
        if hasattr(base, "resampled"):
            return base.resampled(n)
    except (AttributeError, KeyError, TypeError):
        pass
    return plt.get_cmap("turbo", n)


def save_pc1_pc2_kmeans_figure(
    Z: np.ndarray,
    evr: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    k: int,
    n_pc_kmeans: int,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Single PC1 vs PC2 scatter colored by cluster (2D view of space; k-means may use more PCs)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = _discrete_cmap(k)
    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=cluster_labels,
        cmap=cmap,
        vmin=0,
        vmax=max(k - 1, 0),
        alpha=0.55,
        s=12,
        linewidths=0,
    )
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    title = (
        f"PCA projection (K-means K={k}, fitted on first {n_pc_kmeans} PCs)"
        f"{title_suffix}"
    )
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax, label="cluster")
    # avoid crowding tick labels for large K
    if k > 16:
        cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_extra_k_pc_projection_plots(
    Z_full: np.ndarray,
    evr: np.ndarray,
    *,
    n_pc_kmeans: int,
    k_values: list[int],
    seed: int,
    plots_dir: Path,
) -> list[Path]:
    """
    For each K, refit k-means on the same PC subspace and save
    ``pca_pc1_pc2_kmeans_k<K>.png``.
    """
    written: list[Path] = []
    for k in k_values:
        k = int(k)
        if k < 2:
            continue
        _, labels, _ = kmeans_on_pc_block(Z_full, n_pc_kmeans, k, seed)
        if k == 4:
            suffix = " — small K; strong internal silhouette in our grid"
        elif k >= 40:
            suffix = " — large K; NMI vs WWEIA often higher; ARI may stay low"
        else:
            suffix = ""
        out = plots_dir / f"pca_pc1_pc2_kmeans_k{k}.png"
        save_pc1_pc2_kmeans_figure(
            Z_full,
            evr,
            labels,
            k=k,
            n_pc_kmeans=n_pc_kmeans,
            out_path=out,
            title_suffix=suffix,
        )
        written.append(out)
    return written


def save_plots_and_csv(
    result: PcaKMeansResult,
    plots_dir: Path,
    *,
    results_dir: Path | None = None,
    scree_components: int = SCREE_PLOT_COMPONENTS,
) -> None:
    """Write PNGs under ``plots_dir`` and cluster assignments CSV under ``results``."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if results_dir is None:
        results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
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
    fig1.savefig(plots_dir / "pca_scree.png", dpi=150)
    plt.close(fig1)

    save_pc1_pc2_kmeans_figure(
        Z,
        evr,
        result.cluster_labels,
        k=int(km.n_clusters),
        n_pc_kmeans=result.n_pc_kmeans,
        out_path=plots_dir / "pca_pc1_pc2_kmeans.png",
    )

    out_tbl = pd.DataFrame(
        {
            "Food_Name": result.food_names.values,
            "PC1": Z[:, 0],
            "PC2": Z[:, 1],
            "cluster": result.cluster_labels,
        }
    )
    out_tbl.to_csv(results_dir / "unsupervised_kmeans_assignments.csv", index=False)


def save_extra_diagnostic_plots(
    results_dir: Path,
    plots_dir: Path,
    *,
    pca_pc_count: int = 15,
) -> list[Path]:
    """
    Read notebook-exported grid CSVs and write figures that mirror the exploration
    notebook (heatmaps and external-label agreement vs K).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    grid_path = results_dir / "unsupervised_grid_k_m_metrics.csv"
    nopca_path = results_dir / "unsupervised_grid_k_no_pca_metrics.csv"
    wweia_path = results_dir / "unsupervised_k_vs_wweia_nmi_ari.csv"

    if grid_path.exists():
        g = pd.read_csv(grid_path)
        Ks = sorted(g["K"].unique())
        ms = sorted(g["m"].unique())
        mat = np.full((len(Ks), len(ms)), np.nan)
        k_to_i = {k: i for i, k in enumerate(Ks)}
        m_to_j = {m: j for j, m in enumerate(ms)}
        for _, row in g.iterrows():
            mat[k_to_i[int(row["K"])], m_to_j[int(row["m"])]] = float(row["silhouette"])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(ms)))
        ax.set_xticklabels(ms)
        ax.set_yticks(np.arange(len(Ks)))
        ax.set_yticklabels(Ks)
        ax.set_xlabel("m = PCs used for k-means")
        ax.set_ylabel("K = number of clusters")
        ax.set_title("Internal silhouette (PCA + k-means grid)")
        plt.colorbar(im, ax=ax, label="Silhouette", shrink=0.85)
        fig.tight_layout()
        out_hm = plots_dir / "silhouette_heatmap_k_vs_m.png"
        fig.savefig(out_hm, dpi=150)
        plt.close(fig)
        written.append(out_hm)

        if nopca_path.exists():
            nop = pd.read_csv(nopca_path)
            sub = g[g["m"] == float(pca_pc_count)].sort_values("K")
            fig2, ax2 = plt.subplots(figsize=(9, 4.5))
            ax2.plot(
                sub["K"],
                sub["silhouette"],
                marker="o",
                ms=5,
                label=f"PCA + k-means (m = {pca_pc_count})",
            )
            ax2.plot(
                nop["K"],
                nop["silhouette"],
                marker="s",
                ms=5,
                label="k-means on scaled nutrients (no PCA)",
            )
            ax2.set_xlabel("K")
            ax2.set_ylabel("Silhouette")
            ax2.set_title("Internal silhouette vs K: PCA vs no-PCA ablation")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.25)
            fig2.tight_layout()
            out_cmp = plots_dir / "silhouette_vs_k_pca_m15_vs_nopca.png"
            fig2.savefig(out_cmp, dpi=150)
            plt.close(fig2)
            written.append(out_cmp)

    if wweia_path.exists():
        w = pd.read_csv(wweia_path)
        fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax_top.plot(w["K"], w["NMI"], color="C0", marker="o", ms=4)
        ax_top.set_ylabel("NMI")
        ax_top.set_title("Agreement with WWEIA food groups vs K (not ground truth; see report)")
        ax_top.grid(True, alpha=0.25)
        ax_bot.plot(w["K"], w["ARI"], color="C1", marker="s", ms=4)
        ax_bot.set_xlabel("K")
        ax_bot.set_ylabel("ARI")
        ax_bot.grid(True, alpha=0.25)
        fig3.tight_layout()
        out_w = plots_dir / "wweia_nmi_ari_vs_k.png"
        fig3.savefig(out_w, dpi=150)
        plt.close(fig3)
        written.append(out_w)

    return written


def write_run_summary(
    result: PcaKMeansResult,
    *,
    data_path: Path,
    n_features: int,
    k: int,
    pc_kmeans_requested: int,
    seed: int,
    micronutrients_only: bool,
    out_path: Path,
) -> None:
    evr = result.pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    lines = [
        "Aim 1 (unsupervised): PCA + k-means on micronutrient columns",
        "",
        f"Data: {data_path.name}",
        f"Rows: {len(result.food_names)} | Features (numeric): {n_features}",
        f"Predictor set: {'micronutrients only (R/nutrient_definitions.R rules)' if micronutrients_only else 'all numeric columns'}",
        f"K-means: K={k}, PCs used for clustering={result.n_pc_kmeans} (requested {pc_kmeans_requested})",
        f"random_state={seed}",
        f"Inertia: {result.kmeans.inertia_:,.2f}",
        f"Silhouette (on PC space used for k-means): {result.silhouette:.4f}",
        f"Explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}",
        f"Cumulative variance (first {result.n_pc_kmeans} PCs): {cum[result.n_pc_kmeans - 1]:.4f}",
        "",
        "Outputs:",
        f"  plots/unsupervised/pca_scree.png",
        f"  plots/unsupervised/pca_pc1_pc2_kmeans.png",
        "  plots/unsupervised/pca_pc1_pc2_kmeans_k<K>.png (if --also-pc-scatter-k used)",
        f"  results/unsupervised_kmeans_assignments.csv",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA then k-means on USDA nutrient table.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "food_nutrient_conc.csv",
        help="Path to food_nutrient_conc.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR_UNSUPERVISED,
        help="Directory for PNG figures (default: plots/unsupervised/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for CSV assignments and run summary (default: results/)",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K_CLUSTERS)
    parser.add_argument("--pc-kmeans", type=int, default=DEFAULT_N_COMPONENTS_KMEANS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-numeric-nutrients",
        action="store_true",
        help="Use every numeric column in the CSV (include macros and fatty-acid detail). "
        "Default: micronutrient-only columns per R/nutrient_definitions.R.",
    )
    parser.add_argument(
        "--extra-plots",
        action="store_true",
        help="After baseline run, also write figures from grid CSVs in results/ (heatmap, "
        "PCA vs no-PCA silhouette, WWEIA NMI/ARI if those CSVs exist).",
    )
    parser.add_argument(
        "--extra-plots-only",
        action="store_true",
        help="Only generate grid-based figures from existing results/*.csv (skip PCA/k-means).",
    )
    parser.add_argument(
        "--also-pc-scatter-k",
        type=int,
        nargs="*",
        default=[4, 50],
        metavar="K",
        help="Additional K values: write pca_pc1_pc2_kmeans_k<K>.png using the same PCA and m "
        "as the main run (omit extra figures by passing this flag with no values). "
        "Default: 4 50 (compare small-K silhouette choice vs large-K / WWEIA-informed K).",
    )
    args = parser.parse_args()

    if args.extra_plots_only:
        extra = save_extra_diagnostic_plots(args.results_dir, args.plots_dir)
        if not extra:
            print(
                "No extra figures written — expected at least "
                f"{args.results_dir / 'unsupervised_grid_k_m_metrics.csv'}."
            )
        else:
            for p in extra:
                print(f"Wrote {p}")
        return

    X_raw, food_names = load_feature_matrix(
        args.data, micronutrients_only=not args.all_numeric_nutrients
    )
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

    save_plots_and_csv(result, args.plots_dir, results_dir=args.results_dir)

    extra_k = sorted({k for k in args.also_pc_scatter_k if k >= 2 and k != args.k})
    if extra_k:
        for p in save_extra_k_pc_projection_plots(
            result.Z,
            evr,
            n_pc_kmeans=result.n_pc_kmeans,
            k_values=extra_k,
            seed=args.seed,
            plots_dir=args.plots_dir,
        ):
            print(f"Wrote {p}")

    write_run_summary(
        result,
        data_path=args.data,
        n_features=X_raw.shape[1],
        k=args.k,
        pc_kmeans_requested=args.pc_kmeans,
        seed=args.seed,
        micronutrients_only=not args.all_numeric_nutrients,
        out_path=args.results_dir / "unsupervised_aim1_summary.txt",
    )
    print(f"Wrote plots under {args.plots_dir.resolve()}")
    print(f"Wrote assignments + summary under {args.results_dir.resolve()}")
    if args.extra_plots:
        extra = save_extra_diagnostic_plots(args.results_dir, args.plots_dir)
        for p in extra:
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()
