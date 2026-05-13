# Simple EDA Report

## Dataset

This report summarizes exploratory analysis for `food_nutrient_conc.csv` (with label context from `supervised_table.csv`).

## 1) Dataset Structure

- **Foods (rows):** 5431  
- **Unique food names:** 5431  
- **Columns:** 67  
- **Numeric columns (excluding exported blank index):** 65  
- **Character/factor columns:** 1 (`Food_Name`)  
- **Available food labels (from supervised table):** 171 classes

## 2) Missing Data

- **Total missing cells:** 0  
- **Columns with missing values:** 0  
- **Percent missing:** 0.0000%

Interpretation: the dataset is complete and can be modeled directly without NA imputation.

## 3) Variable Distribution Summary

Across numeric variables in `food_nutrient_conc.csv`:

- **Median of variable means:** 2.3799  
- **IQR of variable means:** 27.8763  
- **Median of variable standard deviations:** 4.0804  
- **Median percent zeros (across variables):** 10.9%  
- **Max percent zeros (across variables):** 98.32%  
- **Min percent zeros (across variables):** 0.33%

Interpretation: many nutrient variables are right-skewed and sparse (some are near-zero for most foods), while a few variables are dense.

## 4) Energy (Calories) Distribution

`Energy` in kcal per 100g:

- **Mean:** 199.2517  
- **Median:** 167  
- **SD:** 142.6956  
- **Min:** 0  
- **Max:** 902

Interpretation: energy values are widely dispersed, with a long upper tail (high-calorie foods).

## 5) Class Context for Supervised Tasks

From `supervised_table.csv`:

- **Number of classes:** 171  
- **Largest class size:** 233 foods  
- **Smallest class size:** 1 food

Interpretation: label distribution is imbalanced, with some very rare categories.

## 6) Micronutrient Component Distributions

To better visualize each micronutrient component, two additional plots are generated from `eda.R` using the same micronutrient-only set used in Aim 2 (34 components; macronutrients and detailed fatty-acid variables excluded):

- **Faceted micronutrient histograms (log scale):** `plots/eda/micronutrient_distributions_faceted.png`  
  - One panel per component  
  - **All zero-valued rows are excluded before plotting each component**
  - `log(1 + value)` transformation improves readability for skewed positive values
- **Sparsity bar chart:** `plots/eda/micronutrient_zero_percentage.png`  
  - Shows percent of foods with value equal to zero for each component

Summary from this run:

- **Number of micronutrient components visualized:** 34  
- **Median % zeros across components:** 7.76%  
- **Max % zeros across components:** 96.45%

Interpretation: micronutrient variables have heterogeneous scales and substantial right skew; several components are sparse (many zeros), while others are broadly present across foods.

## 7) Figures and Outputs

- Calorie histogram: `plots/eda/calorie_distribution.png`  
- Micronutrient faceted distributions: `plots/eda/micronutrient_distributions_faceted.png`  
- Micronutrient zero-percentage plot: `plots/eda/micronutrient_zero_percentage.png`  
- Full EDA text output: `results/eda_report.txt`  
- Table 1 (machine-readable): `results/table1_dataset_characteristics.csv`  
- Table 1 (text): `results/table1_dataset_characteristics.txt`

## Overall EDA Takeaway

`food_nutrient_conc.csv` is a clean, fully observed, high-dimensional nutritional dataset with substantial heterogeneity and sparsity across nutrients. Energy has high variance and a heavy upper tail, and supervised labels (171 classes) are imbalanced. These properties support using scaling/regularization and considering robust or nonlinear models in downstream analyses.

See also [`03_unsupervised.md`](03_unsupervised.md) for PCA / k-means on the micronutrient-only feature set.
