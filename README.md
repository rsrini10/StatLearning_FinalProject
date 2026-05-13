# The Nutritional Landscape: Clustering and Mapping Micronutrient Density Across Caloric Profiles

Statistical learning project using the U.S. Department of Agriculture **FoodData Central** survey foods: high-dimensional macronutrient, micronutrient, and caloric information for exploring how nutritional components relate across foods.

Full motivation, aims, and planned evaluation are in [`docs/proposal.md`](docs/proposal.md).

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
├── docs/
│   └── proposal.md              # Project proposal (background, aims, evaluation)
├── unsupervised_learning/       # Python + notebook: PCA / k-means exploration
├── regression/                  # R regression, calorie prediction code
├── results/                     # Text reports and CSV summaries from EDA and classification runs
├── dataprocessing.R             # Builds wide nutrient table and supervised_table from USDA CSVs in data/
├── eda.R                        # EDA on food_nutrient_conc.csv and supervised_table.csv → results/eda_report.txt
├── food_nutrient_conc.csv       # Generated: foods × nutrient concentrations (per 100 g)
└── supervised_table.csv         # Generated: features + labels for supervised / classification work
```

**Data:** Place FoodData Central extracts under `data/` as expected by `dataprocessing.R` (e.g. `food.csv`, `nutrient.csv`, `food_nutrient.csv`, `food_portion.csv`, `survey_fndds_food.csv`, `wweia_food_category.csv`). After running `dataprocessing.R`, use `eda.R` and the scripts in `classification/` and `unsupervised_learning/` as documented in each file.

**Python (unsupervised):** See `unsupervised_learning/requirements.txt` and run notebooks or `unsupervised_learning/unsupervised_pca_kmeans.py` from that directory as needed.

## Team members

— (add names)
