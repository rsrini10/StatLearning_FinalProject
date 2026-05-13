################################################################                
# 1. Setup & Data Loading
################################################################
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)      
library(MASS)       
library(glmnet)     
library(rpart)
library(rpart.plot)
library(xgboost)
library(shapviz)

# Initialize directories
results_dir <- if (dir.exists("results")) "results" else file.path("..", "results")
plot_dir <- file.path("../plots")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Source micronutrient definitions
source("../R/nutrient_definitions.R")

# Load data
dat <- read.csv("../food_nutrient_conc.csv", check.names = FALSE)
if("X" %in% colnames(dat)) dat <- dat %>% dplyr::select(-X)
if("" %in% colnames(dat)) dat <- dat[, colnames(dat) != "", drop = FALSE]

# Define Predictors and Target
y_col <- "Energy"
nm <- names(dat)
pred_names <- micronutrient_predictor_names(nm, y_col = y_col, id_cols = c("Food_Name", "ID", "NDB_No", "fdc_id"))

X <- dat[, pred_names, drop = FALSE]
colnames(X) <- make.names(colnames(X), unique = TRUE)
y <- as.numeric(dat[[y_col]])
  
# Clean NAs
valid_rows <- complete.cases(X) & is.finite(y)
X_clean <- X[valid_rows, , drop = FALSE]
y_clean <- y[valid_rows]

cat(sprintf("Loaded dataset: %d rows, %d predictors.\n", nrow(X_clean), ncol(X_clean)))

################################################################
# 2. Train/Test Split & Preprocessing
################################################################
set.seed(42)
test_idx <- sample.int(nrow(X_clean), size = floor(0.2 * nrow(X_clean)))
train_idx <- setdiff(seq_len(nrow(X_clean)), test_idx)

X_train <- X_clean[train_idx, ]
X_test  <- X_clean[test_idx, ]
y_train <- y_clean[train_idx]
y_test  <- y_clean[test_idx]

# Standardize for Linear Models (Elastic Net/LDA)
preProc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProc, X_train)
X_test_scaled  <- predict(preProc, X_test)

# Prepare specific formats
X_train_mat <- as.matrix(X_train_scaled)
X_test_mat  <- as.matrix(X_test_scaled)
train_tree_df <- cbind(Energy = y_train, as.data.frame(X_train))
test_tree_df  <- cbind(Energy = y_test, as.data.frame(X_test))
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

################################################################
# 3. LDA Exploration (Non-Bayesian)
################################################################

cat("\n--- Running LDA Exploration ---\n")

make_tertile_breaks <- function(values) {
  breaks <- as.numeric(quantile(values, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE))
  if (anyDuplicated(breaks)) {
    breaks <- breaks + c(0, 1e-08, 2e-08, 3e-08)
    breaks[1] <- min(values, na.rm = TRUE)
    breaks[length(breaks)] <- max(values, na.rm = TRUE)
  }
  breaks
}

tertile_breaks <- make_tertile_breaks(y_train)
y_train_binned <- cut(y_train, breaks = tertile_breaks, labels = c("Low", "Medium", "High"), include.lowest = TRUE)
y_test_binned  <- cut(y_test, breaks = tertile_breaks, labels = c("Low", "Medium", "High"), include.lowest = TRUE)

lda_fit <- lda(x = X_train_scaled, grouping = y_train_binned)
lda_preds <- predict(lda_fit, X_test_scaled)
lda_weights <- abs(lda_fit$scaling[, "LD1"])
lda_imp_df  <- data.frame(Micronutrient = names(lda_weights), LD1_Importance = as.numeric(lda_weights)) %>% arrange(desc(LD1_Importance))

# Save LDA Plot
p_lda <- ggplot(head(lda_imp_df, 15), aes(x = reorder(Micronutrient, LD1_Importance), y = LD1_Importance)) +
  geom_col(fill = "steelblue") + coord_flip() + theme_minimal() +
  labs(title = "Top 15 Drivers of Caloric Density (LDA)", x = "Micronutrient", y = "Weight")
ggsave(file.path(plot_dir, "01_LDA_Importance.png"), p_lda)

################################################################
# 4. Grid Search: Elastic Net
################################################################
cat("\n--- Tuning Elastic Net via Grid Search ---\n")

# Define the grid
en_grid <- expand.grid(
  alpha = seq(0, 1, length.out = 11),  # Check every 0.1 from 0 (Ridge) to 1 (Lasso)
  lambda = seq(0.0001, 0.1, length.out = 10)
)

en_ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

en_train_mod <- train(
  x = X_train_mat, y = y_train,
  method = "glmnet",
  metric = "RMSE",
  tuneGrid = en_grid,
  trControl = en_ctrl
)

# Extract best params and final model
best_en <- en_train_mod$bestTune
en_final <- en_train_mod$finalModel
en_preds <- as.numeric(predict(en_train_mod, X_test_mat))
en_coefs <- as.matrix(coef(en_final, s = best_en$lambda))
en_nonzero_terms <- sum(as.numeric(en_coefs[-1, 1]) != 0)

cat(sprintf("Best Elastic Net: alpha = %.2f, lambda = %.4f\n", best_en$alpha, best_en$lambda))

################################################################
# 5. Grid Search: Decision Tree
################################################################
cat("\n--- Tuning Decision Tree via Grid Search ---\n")

# For rpart, caret typically tunes 'cp'
tree_grid <- expand.grid(cp = seq(0, 0.05, length.out = 20))
tree_maxdepth <- 5

tree_train_mod <- train(
  x = X_train, y = y_train,
  method = "rpart",
  metric = "RMSE",
  tuneGrid = tree_grid,
  # Pass maxdepth via the control argument
  control = rpart.control(maxdepth = tree_maxdepth),
  trControl = trainControl(method = "cv", number = 5)
)

best_tree <- tree_train_mod$bestTune
tree_fit <- tree_train_mod$finalModel
tree_preds <- as.numeric(predict(tree_train_mod, X_test))

# Plot the tree
png(file.path(plot_dir, "02_Decision_Tree.png"), width = 2000, height = 1500, res = 300)
rpart.plot(tree_fit, main = "Optimized Decision Tree (Grid Search)", box.palette = "RdYlGn")
dev.off()

################################################################
# 6. Grid Search: XGBoost
################################################################
cat("\n--- Tuning XGBoost via Grid Search ---\n")

# Grid search for XGBoost can be slow; this is a focused grid
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(2, 4, 6),
  eta = c(0.01, 0.1, 0.3),
  gamma = 0,
  colsample_bytree = c(0.6, 0.8),
  min_child_weight = 1,
  subsample = c(0.7, 1)
)

xgb_train_mod <- train(
  x = as.matrix(X_train), y = y_train,
  method = "xgbTree",
  metric = "RMSE",
  tuneGrid = xgb_grid,
  trControl = trainControl(method = "cv", number = 5),
  verbosity = 0
)

best_xgb <- xgb_train_mod$bestTune
xgb_fit  <- xgb_train_mod$finalModel
xgb_preds <- as.numeric(predict(xgb_train_mod, as.matrix(X_test)))

# Prepare SHAP analysis (requires the xgb.Booster object)
shp <- shapviz(xgb_fit, X_pred = as.matrix(X_train))

adjusted_r2 <- function(actual, predicted, p) {
  n <- length(actual)
  if (n <= p + 1) {
    return(NA_real_)
  }
  r2 <- cor(actual, predicted)^2
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

################################################################
# 7. Final Report & Metrics (Updated for Grid Search)
################################################################
# Use the same results_comparison logic as before
results_comparison <- data.frame(
  Actual = rep(y_test, 3),
  Predicted = c(en_preds, tree_preds, xgb_preds),
  Model = rep(c("Elastic Net", "Decision Tree", "XGBoost"), each = length(y_test))
)

################################################################
# 3. Importance Analysis & Plots
################################################################
cat("\n--- Producing Importance Plots ---\n")

# --- 7a. Elastic Net Coefficient Importance ---
en_imp_df <- data.frame(
  Feature = rownames(en_coefs),
  Coef = as.numeric(en_coefs),
  stringsAsFactors = FALSE
)
en_imp_df <- en_imp_df[en_imp_df$Feature != "(Intercept)" & en_imp_df$Coef != 0, , drop = FALSE]
en_imp_df$AbsCoef <- abs(en_imp_df$Coef)
en_imp_df <- en_imp_df[order(-en_imp_df$AbsCoef), , drop = FALSE]
en_imp_df <- head(en_imp_df, 15)

p_en_imp <- ggplot(en_imp_df, aes(x = reorder(Feature, AbsCoef), y = Coef, fill = Coef > 0)) +
  geom_col() + 
  coord_flip() +
  scale_fill_manual(values = c("firebrick", "steelblue"), labels = c("Negative", "Positive")) +
  labs(title = "Elastic Net: Top 15 Predictors", 
       subtitle = "Standardized coefficients (Higher magnitude = stronger impact)",
       x = "Micronutrient", y = "Coefficient Value", fill = "Direction of Impact") +
  theme_minimal()

ggsave(file.path(plot_dir, "03_ElasticNet_Importance_GridSearch.png"), p_en_imp, width = 8, height = 6)


# --- 7b. Decision Tree Structure ---
# --- 7c. XGBoost SHAP Summary ---
# SHAP provides the most nuanced view of importance for non-linear models
shp <- shapviz(xgb_fit, X_pred = as.matrix(X_train))
p_shap <- sv_importance(shp, kind = "beeswarm") + 
  labs(title = "XGBoost SHAP: Micronutrient Impact",
       subtitle = "Shows how high/low concentrations push caloric prediction up or down") +
  theme_minimal()

ggsave(file.path(plot_dir, "03_XGBoost_SHAP_Importance.png"), p_shap, width = 10, height = 8)


################################################################
# 8. Aim 2: Performance Comparison (Final Summary)
################################################################
cat("\n--- Generating Individual Performance Plots for Aim 2 ---\n")

# 1. Create the combined results dataframe
results_comparison <- data.frame(
  Actual = rep(y_test, 3),
  Predicted = c(en_preds, tree_preds, xgb_preds),
  Model = rep(c("Elastic Net", "Decision Tree", "XGBoost"), each = length(y_test))
)

# 2. Loop through each model and save an individual plot
models <- unique(results_comparison$Model)

for (m in models) {
  # Subset data for the specific model
  model_data <- results_comparison %>% filter(Model == m)
  
  # Calculate specific metrics for this model's subtitle
  m_rmse <- sqrt(mean((model_data$Actual - model_data$Predicted)^2))
  m_adj_r2 <- switch(
    m,
    "Elastic Net" = adjusted_r2(model_data$Actual, model_data$Predicted, en_nonzero_terms),
    "Decision Tree" = adjusted_r2(model_data$Actual, model_data$Predicted, sum(tree_fit$frame$var != "<leaf>")),
    "XGBoost" = adjusted_r2(model_data$Actual, model_data$Predicted, nrow(xgb.importance(model = xgb_fit))),
    NA_real_
  )
  
  # Clean filename (replace spaces with underscores)
  clean_name <- gsub(" ", "_", m)
  file_name <- sprintf("04_Aim2_Actual_vs_Predicted_%s.png", clean_name)
  
  # Create individual plot
  p <- ggplot(model_data, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.3, color = "steelblue") +
    geom_abline(intercept = 0, slope = 1, color = "darkred", linetype = "dashed", linewidth = 0.8) +
    labs(
      title = paste("Aim 2: Model Reliability -", m),
      subtitle = sprintf("RMSE: %.2f | Adj. R^2: %.3f", m_rmse, m_adj_r2),
      x = "Actual Energy (kcal)", 
      y = "Predicted Energy (kcal)"
    ) +
    theme_minimal() +
    coord_fixed(xlim = c(0, max(y_test)), ylim = c(0, max(y_test)))
  
  # Save the plot
  ggsave(file.path(plot_dir, file_name), plot = p, width = 7, height = 7, dpi = 300, bg = "white")
  
  cat(sprintf("Saved: %s\n", file_name))
}

# 3. Final metrics printout for reference
metrics <- results_comparison %>%
  dplyr::group_by(Model) %>%
  dplyr::summarise(
    RMSE = sqrt(mean((Actual - Predicted)^2)),
    .groups = "drop"
  ) %>%
  dplyr::arrange(RMSE)

test_rmse_metrics <- metrics %>%
  dplyr::select(Model, RMSE)

cat("\nAnalysis Complete. Summary Metrics:\n")
print(metrics)
################################################################
# 9. LDA Classification Metrics & Confusion Matrix
################################################################
cat("\n--- Computing LDA Performance Metrics ---\n")

# Compute metrics using caret's confusionMatrix
# Ensure factors have the same levels
lda_cm <- confusionMatrix(lda_preds$class, y_test_binned)

# Extract specific metrics
lda_accuracy <- lda_cm$overall["Accuracy"]
# F1 is usually provided per-class in multiclass; we can take the macro-average
lda_f1_macro <- mean(lda_cm$byClass[, "F1"], na.rm = TRUE)

# Save the Confusion Matrix as a CSV for easy reference
write.csv(as.data.frame(lda_cm$table), 
          file.path(results_dir, "lda_confusion_matrix.csv"), row.names = FALSE)

################################################################
# 10. Final Report Generation
################################################################
cat("\n--- Saving Final Report to Results Folder ---\n")
test_rmse_file <- file.path(results_dir, "ENet_Trees_XG_test_rmse_metrics.csv")
write.csv(test_rmse_metrics, test_rmse_file, row.names = FALSE)

cat(sprintf("Report saved to: %s\n", report_file))
cat(sprintf("Confusion matrix saved to: %s\n", file.path(results_dir, "lda_confusion_matrix.csv")))
cat(sprintf("Test RMSE metrics saved to: %s\n", test_rmse_file))
