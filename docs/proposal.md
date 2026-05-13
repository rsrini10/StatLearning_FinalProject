# The Nutritional Landscape: Clustering and Mapping Micronutrient Density Across Caloric Profiles in USDA FoodData Central

## Background

It is vital that people eat nutritious food. By performing unsupervised methods, we can obtain subgroups of foods that have similar mineral and nutrient contents, and this information can be used to construct a balanced diet where one can evaluate which foods to eat to obtain enough nutrients from each group. In addition, we plan to predict the caloric content of foods from their micronutrients. There has not been extensive research on the use of tree-based methods for evaluating the relationship between micronutrients and caloric content, so we aim to help address this gap in the literature.

## Dataset

We will use the U.S. Department of Agriculture (USDA) Survey Foods dataset (FoodData Central). Each entry contains detailed information on each food’s macronutrients (carbohydrates, proteins, fats), micronutrients (vitamins, minerals), and caloric content. The dataset is high-dimensional and granular, making it well suited for exploring relationships between nutritional components across different foods.

## Objectives

This project aims to explore relationships between nutrition components in foods people eat.

- **Primary objective:** Learn patterns in everyday foods so we can support more precise recommendations, especially for people with dietary restrictions.
- **Secondary objective:** Learn clustering patterns of nutritional profiles for each menu (or food context). Using these clusters, recommendations can be based on a **nutrition profile cluster** rather than on a single specific menu item.

## Methods (overview)

We will first use **unsupervised** methods to group food types by nutrient and mineral composition. We will then use **supervised regression** models to predict the caloric content of foods from nutrient and mineral concentrations.

---

## Aim 1: Unsupervised clustering of foods by micronutrient profile

Group similar foods based on micronutrient levels using methods such as **PCA**, **k-means clustering**, and/or **hierarchical clustering**. To inform the choice of the number of clusters, we will use the **elbow method** by evaluating **within-cluster sum of squares (WCSS)** and related internal criteria as appropriate.

## Aim 2: Supervised prediction of caloric content from micronutrients

Predict caloric content from micronutrient levels using regression approaches, including:

- **LASSO** (for shrinkage and feature selection),
- **Tree-based methods** (e.g., random forest, gradient boosting),
- **Discriminant analysis** (where applicable in the modeling pipeline),
- **KNN regression**.

We will use **grid search** or **Bayesian optimization** to tune hyperparameters with respect to **validation-set MSE** on caloric content (with **k-fold cross-validation** on the training portion as needed).

---

## Evaluation

**Clustering (Aim 1):** We will assess cluster structure using internal validation measures, including **within-cluster sum of squares (WCSS)** and related diagnostics, to support decisions such as the number of clusters (e.g., in conjunction with the elbow plot and stability checks where feasible).

**Regression (Aim 2):** We will split the data into **training and test** sets, use **k-fold cross-validation on the training set** for hyperparameter tuning, and compare models using standard predictive metrics, including **RMSE**, **MAE**, and **R²**, to identify which approach best captures the relationship between micronutrient composition and caloric content.
