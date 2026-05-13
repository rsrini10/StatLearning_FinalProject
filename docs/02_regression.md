# 02 Regression: Aim 2 Results

## Goal

Predict food caloric content (`Energy`, kcal/100g) from **micronutrient composition**.

The core idea is to evaluate how much caloric variation is explained by vitamins/minerals-related signals when direct macronutrient drivers are excluded.

## Data and Features

- **Source dataset:** `food_nutrient_conc.csv`
- **Rows used:** 5431 foods
- **Target:** `Energy`
- **Predictors:** 34 micronutrient-related numeric variables
- **Train/test split:** 80/20 random split
  - Train: 4345
  - Test: 1086

### Excluded predictor groups

To avoid trivial calorie prediction from macros, the following were excluded:

- `Total lipid (fat)`, `Total Sugars`, `Carbohydrate, by difference`, `Protein`, `Water`, `Fiber, total dietary`, `Alcohol, ethyl`, `Cholesterol`
- `Fatty acids, total saturated`, `Fatty acids, total monounsaturated`, `Fatty acids, total polyunsaturated`
- all detailed fatty-acid chain columns matching prefixes: `SFA`, `MUFA`, `PUFA`

## Models Compared

- OLS linear regression
- Ridge regression (`glmnet`, CV-selected lambda)
- LASSO regression (`glmnet`, CV-selected lambda)
- Regression tree (`rpart`), with `cp`, `minbucket`, and `minsplit` selected by 5-fold CV grid search
- Random forest regression (`randomForest`), with **`mtry`** and **`ntree`** selected by a 5-fold CV grid (factorial grid; see script for values)
- k-NN regression (`FNN::knn.reg`), with `k` selected by 5-fold CV on training data

## Evaluation metrics

- **RMSE** (primary headline metric; lower is better)
- **MAE** (lower is better)

Scaling and CV are documented in `regression/predict_calories.R` and `results/regression_aim2_model_selection_methods.txt` (per-fold z-scores for CV where applicable; glmnet on raw `x` with `standardize=TRUE`).

All metrics below are on the held-out **test set**. For an up-to-date leaderboard after code changes, run `Rscript regression/predict_calories.R` and read `results/regression_aim2_metrics.txt`.

## Results

| Model | RMSE | MAE |
|---|---:|---:|
| Random forest (5-fold CV, mtry=9, ntree=400) | 49.5846 | 29.0598 |
| k-NN regression (k=3, 5-fold CV) | 59.7932 | 33.2317 |
| Regression tree (rpart, 5-fold CV) | 72.7776 | 42.2042 |
| LASSO (glmnet, 5-fold CV) | 163.3312 | 82.6815 |
| Ridge (glmnet, 5-fold CV) | 259.9108 | 88.7445 |
| OLS (micronutrients only) | 413.2599 | 91.4065 |

*Values from the latest `Rscript regression/predict_calories.R` run (test set; `results/regression_aim2_metrics.txt`). Tuning snapshot: rpart best `cp` = 1e-04, `minbucket` = 3, `minsplit` = 20; RF grid over `mtry` × `ntree` with chosen pair (9, 400); k-NN best `k` = 3.*

## Interpretation

1. **Nonlinear models strongly outperform linear models.**  
   Random forest and k-NN capture nonlinear structure in micronutrient-energy relationships much better than OLS/ridge/lasso.

2. **Random forest is the best model in this experiment.**  
   It achieved the lowest test RMSE among the candidates compared.

3. **Linear models underfit this task under current feature restrictions.**  
   OLS/ridge/lasso show much higher test RMSE than the nonlinear methods under micronutrient-only inputs.

4. **Micronutrients alone still carry substantial predictive signal.**  
   Even without direct macronutrient variables, the best model explains a large portion of energy variance.

## Artifacts Produced

- Metrics: `results/regression_aim2_metrics.txt`
- Model summaries: `results/regression_aim2_model_summaries.txt`
- **Supplementary hyperparameter / grid search tables** (written when you run `regression/predict_calories.R`):
  - `results/supplementary_aim2_hyperparameter_summary.csv` — one row per method + global split/CV rules
  - `results/supplementary_aim2_hyperparameter_rpart_grid.csv` — full rpart `cp` × `minbucket` × `minsplit` grid with CV RMSE
  - `results/supplementary_aim2_hyperparameter_rf_grid.csv` — `mtry` × `ntree` factorial grid with mean CV RMSE
  - `results/supplementary_aim2_hyperparameter_knn_k.csv` — `k` vs mean CV RMSE
  - `results/supplementary_aim2_hyperparameter_grids.txt` — tab-separated copy of all tables for quick paste
- Prediction plots:
  - `plots/regression_aim2_actual_vs_predicted_grid.png`
  - `plots/regression_aim2_actual_vs_predicted_best.png`

## Notes and Next Steps

- Current split is a single random train/test split. For more stable reporting, add repeated CV or repeated holdout.
- Consider hyperparameter tuning expansions for RF/k-NN and adding boosted trees.
- To support the “micronutrient-per-calorie” framing directly, add a derived density score analysis (e.g., standardized micronutrient score divided by calories).
