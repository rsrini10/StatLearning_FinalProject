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
- Regression tree (`rpart`)
- Random forest regression (`randomForest`)
- k-NN regression (`FNN::knn.reg`), with `k` selected by 5-fold CV on training data

## Evaluation Metrics

- **RMSE** (lower is better)
- **MAE** (lower is better)
- **R²** (higher is better)

All reported below are on the held-out **test set**.

## Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random forest | 50.9090 | 30.2372 | 0.8819 |
| k-NN regression (k=3) | 59.7932 | 33.2317 | 0.8371 |
| Regression tree (rpart) | 75.0360 | 45.6572 | 0.7434 |
| LASSO (glmnet) | 180.5559 | 83.3905 | -0.4855 |
| Ridge (glmnet) | 252.7258 | 88.7366 | -1.9104 |
| OLS (micronutrients only) | 413.2599 | 91.4065 | -6.7822 |

## Interpretation

1. **Nonlinear models strongly outperform linear models.**  
   Random forest and k-NN capture nonlinear structure in micronutrient-energy relationships much better than OLS/ridge/lasso.

2. **Random forest is the best model in this experiment.**  
   It achieved the lowest RMSE/MAE and highest R².

3. **Linear models underfit this task under current feature restrictions.**  
   Negative test R² for OLS/ridge/lasso indicates poor generalization relative to a mean-prediction baseline.

4. **Micronutrients alone still carry substantial predictive signal.**  
   Even without direct macronutrient variables, the best model explains a large portion of energy variance.

## Artifacts Produced

- Metrics: `results/regression_aim2_metrics.txt`
- Model summaries: `results/regression_aim2_model_summaries.txt`
- Prediction plots:
  - `plots/regression_aim2_actual_vs_predicted_grid.png`
  - `plots/regression_aim2_actual_vs_predicted_best.png`

## Notes and Next Steps

- Current split is a single random train/test split. For more stable reporting, add repeated CV or repeated holdout.
- Consider hyperparameter tuning expansions for RF/k-NN and adding boosted trees.
- To support the “micronutrient-per-calorie” framing directly, add a derived density score analysis (e.g., standardized micronutrient score divided by calories).
