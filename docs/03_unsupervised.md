# Aim 1 — Unsupervised clustering (micronutrient profile)

## Aim

Discover **structure in USDA foods** using only **micronutrient composition** (vitamins, minerals, carotenoids, choline, etc.). Macronutrients and long-chain fatty-acid detail are excluded so the representation matches the predictor set used elsewhere in this project (`R/nutrient_definitions.R`, mirrored in `unsupervised_learning/nutrient_definitions.py`). The goal is **data-driven groups** in standardized nutrient space, not to recover any particular external labeling scheme.

## Plan

1. **Preprocess** each retained column: median imputation (rarely needed here), then z-scoring.
2. **Reduce dimension** with PCA on all retained components; run **k-means** on the leading *m* PC scores (baseline *m* = 15, *K* = 8).
3. **Tune** *K* and *m* over a grid in `pca_kmeans_exploration.ipynb`, record internal criteria and (optionally) compare to **k-means without PCA** on the same scaled features.
4. **External sanity check only:** overlap between cluster labels and **WWEIA / FNDDS food-group** wording is summarized with **NMI** and **ARI** across *K*. This is **not** ground truth: WWEIA reflects administrative menu categories; our inputs are **concentration-style micronutrients**, so weak agreement is expected and does not invalidate the clusters.

## Metrics (how to read them)

| Metric | Role | Interpretation |
|--------|------|----------------|
| **Silhouette** (on the space used for k-means, e.g. first *m* PCs) | Internal cluster quality | Higher means points are relatively closer to their own cluster than to others. Useful for comparing *K* / *m* or PCA vs no-PCA, not a “truth” score. |
| **Inertia** (within-cluster sum of squares to centroids) | Internal fit | Decreases as *K* increases; use together with silhouette and stability, not alone. |
| **Calinski–Harabasz** | Internal separation | Variance ratio between and within clusters; exported in grid tables from the notebook. |
| **NMI** (normalized mutual information) | External overlap (optional) | Measures **information shared** between cluster labels and an external label vector, normalized to ~\[0, 1\]. Sensitive to **number of clusters** vs number of external groups; high NMI does not mean the two taxonomies are “the same.” |
| **ARI** (adjusted Rand index) | External overlap (optional) | Agreement vs **chance**, adjusted for label cardinality; range roughly \[-1, 1\] with 0 ≈ random. **Low ARI with moderate NMI** often appears when cluster counts and group sizes differ (many small clusters vs fewer coarse groups). |

**Important caveat for WWEIA / FNDDS:** those labels describe **how foods are categorized on surveys and menus**, not a biochemical ground truth. Our clusters summarize **micronutrient concentration profiles**. Reporting **weak or mixed** NMI/ARI supports the honest conclusion that nutrient-based structure aligns only loosely with those wording-based groups.

## Code and notebook

| Artifact | Purpose |
|----------|---------|
| `unsupervised_learning/unsupervised_pca_kmeans.py` | CLI: baseline figures, assignments CSV, optional **grid-based** figures (`--extra-plots` or `--extra-plots-only`) |
| `unsupervised_learning/pca_kmeans_exploration.ipynb` | Narrative, **(K, m)** grid, no-PCA ablation, PC loadings, WWEIA comparison exports |

Run the script from `unsupervised_learning/`:

```bash
python unsupervised_pca_kmeans.py
python unsupervised_pca_kmeans.py --extra-plots          # after grid CSVs exist
python unsupervised_pca_kmeans.py --extra-plots-only       # refresh figures from results/*.csv only
```

Options: `--k`, `--pc-kmeans`, `--seed`, `--all-numeric-nutrients`, `--plots-dir`, `--results-dir`.

## Outputs (repository convention)

**Figures** — `plots/unsupervised/`

- `pca_scree.png` — explained variance for leading PCs (baseline run)  
- `pca_pc1_pc2_kmeans.png` — PC1 vs PC2 colored by baseline cluster  
- `silhouette_heatmap_k_vs_m.png` — internal silhouette over the *(K, m)* grid (`--extra-plots`)  
- `silhouette_vs_k_pca_m15_vs_nopca.png` — PCA (*m* = 15) vs **no PCA** silhouette vs *K* (`--extra-plots`)  
- `wweia_nmi_ari_vs_k.png` — **NMI** and **ARI** vs *K* vs WWEIA groups (`--extra-plots`, requires CSV below)

**Tables / text** — `results/`

- `unsupervised_kmeans_assignments.csv` — `Food_Name`, PC1, PC2, cluster (baseline)  
- `unsupervised_aim1_summary.txt` — run metadata and metrics (baseline CLI)  
- `unsupervised_grid_k_m_metrics.csv` — silhouette, inertia, Calinski–Harabasz over *K* and *m* (notebook export)  
- `unsupervised_grid_k_no_pca_metrics.csv` — same metrics without PCA (notebook export)  
- `unsupervised_k_vs_wweia_nmi_ari.csv` — *K* vs NMI/ARI vs WWEIA (notebook export, optional)

## Findings (how we interpret results)

- **Internal structure:** Silhouette in PC space often favors **small *K*** and **moderate *m*** in our grids; the heatmap shows sensitivity to both knobs. The **no-PCA** curve illustrates whether compression helps or hurts separation for a given *K*.  
- **Projection plots (PC1–PC2):** k-means may use **many** PCs; two-dimensional scatter is only a **view** of the fitted space.  
- **Cluster sizes:** k-means can yield **imbalanced** clusters (e.g. one large “typical” blob); interpret with profiles of cluster means in nutrient space (notebook).  
- **WWEIA / NMI–ARI:** We use these as a **sanity check**, not validation. Nutrient-based clusters need not match menu food-group wording; **low ARI** in particular is consistent with different **granularity** and **label semantics** rather than “failure” of clustering.

## How to regenerate everything

1. Ensure `food_nutrient_conc.csv` exists at the project root.  
2. `cd unsupervised_learning && python unsupervised_pca_kmeans.py --extra-plots`  
3. Open `pca_kmeans_exploration.ipynb` and run all cells (refreshes grid and WWEIA CSVs if you change settings).  
4. If you only changed exported CSVs, `python unsupervised_pca_kmeans.py --extra-plots-only` updates the grid-based PNGs.

EDA figures for micronutrient distributions live under **`plots/eda/`** (from `eda.R`); regression figures under **`plots/regression/`** (from `regression/predict_calories.R`).
