# The Nutritional Landscape: Clustering and Mapping Micronutrient Density Across Caloric Profiles

Statistical learning project using the U.S. Department of Agriculture **FoodData Central** survey foods: high-dimensional macronutrient, micronutrient, and caloric information for exploring how nutritional components relate across foods.

Full motivation, aims, and planned evaluation are in [`docs/00_proposal.md`](docs/00_proposal.md).  
Write-ups: [`docs/01_simple_eda_report.md`](docs/01_simple_eda_report.md), [`docs/02_regression.md`](docs/02_regression.md), [`docs/03_unsupervised.md`](docs/03_unsupervised.md).

## Objectives

- **Primary:** Learn patterns in everyday foods to support more precise recommendations, especially for people with dietary restrictions.
- **Secondary:** Discover clustering of nutritional profiles by menu or food context so recommendations can target a **nutrition profile cluster** rather than a single menu item.

## Planned methods (from proposal)

| Aim | Focus | Methods |
|-----|--------|---------|
| **1 — Unsupervised** | Group foods by micronutrient profile | PCA, k-means and/or hierarchical clustering; elbow / WCSS and related diagnostics for *k* |
| **2 — Supervised** | Predict calories from micronutrients | LASSO, tree-based models (e.g. random forest, boosting), discriminant analysis where applicable, KNN regression; tuning via grid search or Bayesian optimization with k-fold CV; compare RMSE, MAE, R² |

This repository already includes exploratory analysis, unsupervised prototypes, and **classification** work on food categories (WWEIA coarse groupings, KNN, multinomial logistic regression) built on the supervised table.

## Repository layout

```
StatLearning_FinalProject/
├── R/
│   └── nutrient_definitions.R   # Macro vs micronutrient column rules (shared with regression / eda.R)
├── docs/
│   ├── 00_proposal.md
│   ├── 01_simple_eda_report.md
│   ├── 02_regression.md
│   └── 03_unsupervised.md       # Aim 1: PCA / k-means, outputs layout
├── plots/                       # Figures (grouped by aim)
│   ├── eda/                     # From eda.R (distributions, micronutrient panels)
│   ├── regression/             # From regression/predict_calories.R
│   └── unsupervised/           # From unsupervised_pca_kmeans.py / notebook
├── unsupervised_learning/       # Python + notebook; nutrient_definitions.py mirrors R column rules
├── regression/                  # R experiment scripts (e.g. predict_calories.R) → results/ + plots/regression/
├── results/                     # Text reports and CSVs (EDA, regression, unsupervised)
├── dataprocessing.R             # Builds food_nutrient_conc.csv and supervised outputs from data/ CSVs
├── eda.R                        # EDA → results/eda_report.txt, plots/eda/
├── food_nutrient_conc.csv       # Generated: foods × nutrient concentrations (per 100 g)
└── supervised_table.csv         # Features + labels for supervised work
```

**Data:** Place FoodData Central extracts under `data/` as expected by `dataprocessing.R`. After building the wide tables, run `eda.R`, `regression/predict_calories.R`, and `unsupervised_learning/` as documented in each file.

**Python (unsupervised):** See `unsupervised_learning/requirements.txt`. Run `python unsupervised_pca_kmeans.py` from `unsupervised_learning/` (writes `plots/unsupervised/` and `results/`).

## Team members

— (add names)
