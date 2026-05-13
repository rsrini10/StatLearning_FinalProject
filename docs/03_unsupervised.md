# Aim 1 — Unsupervised clustering (micronutrient profile)

## Goal

Group USDA survey foods by **micronutrient composition** (vitamins, minerals, carotenoids, choline, etc.) after excluding macronutrients and detailed fatty-acid chain columns. The rules match `R/nutrient_definitions.R` and are implemented in Python as `unsupervised_learning/nutrient_definitions.py`.

## Methods (baseline)

1. Read `food_nutrient_conc.csv` (project root).
2. **Median imputation** (not usually needed; data are complete) and **standardize** each retained nutrient column.
3. **PCA** on all retained components; **k-means** on the first *m* principal component scores (default *m* = 15, default *K* = 8 clusters).
4. Diagnostics: **silhouette** on the PC subspace used for k-means, **inertia**, variance explained.

## Code and notebook

| Artifact | Purpose |
|----------|---------|
| `unsupervised_learning/unsupervised_pca_kmeans.py` | CLI: reproduce baseline figures and `results/` tables |
| `unsupervised_learning/pca_kmeans_exploration.ipynb` | Narrative + baseline + **(K, m)** grid, no-PCA ablation, PC loadings |

Run the script from `unsupervised_learning/`:

```bash
python unsupervised_pca_kmeans.py
```

Options: `--k`, `--pc-kmeans`, `--seed`, `--all-numeric-nutrients` (use full numeric table instead of micronutrient-only), `--plots-dir`, `--results-dir`.

## Outputs (repository convention)

**Figures** — `plots/unsupervised/`

- `pca_scree.png` — explained variance for leading PCs  
- `pca_pc1_pc2_kmeans.png` — PC1 vs PC2 colored by cluster  

**Tables / text** — `results/`

- `unsupervised_kmeans_assignments.csv` — `Food_Name`, PC1, PC2, cluster ID  
- `unsupervised_aim1_summary.txt` — run metadata and metrics (from CLI)  
- `unsupervised_grid_k_m_metrics.csv` — silhouette, inertia, Calinski–Harabasz over a grid of *K* and *m* (from notebook export cell, if run)  
- `unsupervised_grid_k_no_pca_metrics.csv` — same metrics for k-means on scaled micronutrients **without** PCA  

## How to regenerate everything

1. Ensure `food_nutrient_conc.csv` exists at the project root.  
2. `cd unsupervised_learning && python unsupervised_pca_kmeans.py`  
3. Open `pca_kmeans_exploration.ipynb` and run all cells (refresh grid CSVs if you change the grid).  

EDA figures for micronutrient distributions live under **`plots/eda/`** (from `eda.R`); regression figures under **`plots/regression/`** (from `regression/predict_calories.R`).

## Interpretation notes

- Cluster sizes are often **imbalanced** (one dominant “typical foods” blob); combine with grid search over *K* and *m* and qualitative cluster profiles.  
- Silhouette is only one internal criterion; see the notebook for heatmaps and the no-PCA comparison.  
- PC1–PC2 plots are **projections**; k-means may use more than two PCs.
