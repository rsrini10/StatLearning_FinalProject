# Aim 2: Predict Energy (kcal/100g) from *micronutrient* composition only (regression).
#
# Motivation: Macros (fat, carbohydrate, protein, water, fiber, alcohol) largely
# determine calories (~4/4/9 kcal/g). Excluding them lets the models use vitamins,
# minerals, and related compounds so you can study how well "micronutrient profile"
# aligns with energy — a step toward comparing foods on micronutrient delivery
# relative to calories (e.g. residuals or separate density indices).
#
# Methods: OLS, ridge & LASSO (glmnet::cv.glmnet), rpart, randomForest, FNN::knn.reg.
# Discriminant analysis does not apply to continuous Energy; omitted.
#
# Run from project root:
#   Rscript regression/predict_calories.R
# Writes metrics, model summaries, and PNGs to results/ and plots/.
#
# Requires: glmnet, randomForest, FNN, rpart

suppressPackageStartupMessages({
  library(glmnet)
  library(randomForest)
  library(FNN)
  library(rpart)
})

csv <- if (file.exists("food_nutrient_conc.csv")) {
  "food_nutrient_conc.csv"
} else if (file.exists(file.path("..", "food_nutrient_conc.csv"))) {
  file.path("..", "food_nutrient_conc.csv")
} else {
  stop("Cannot find food_nutrient_conc.csv (run from project root or regression/).")
}

results_dir <- if (dir.exists("results")) "results" else file.path("..", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
out_metrics <- file.path(results_dir, "regression_aim2_metrics.txt")
out_summaries <- file.path(results_dir, "regression_aim2_model_summaries.txt")

dat <- read.csv(csv, check.names = FALSE, stringsAsFactors = FALSE)
y_col <- "Energy"
stopifnot(y_col %in% names(dat))

nm <- names(dat)
drop_id <- nm %in% c("", "Food_Name", y_col)

# Exclude macronutrients and detailed fatty acids so predictors reflect
# micronutrients + related non-macro compounds (vitamins, minerals, carotenoids, etc.).
macro_exact <- c(
  "Total lipid (fat)",
  "Total Sugars",
  "Carbohydrate, by difference",
  "Protein",
  "Water",
  "Fiber, total dietary",
  "Alcohol, ethyl",
  "Cholesterol",
  "Fatty acids, total saturated",
  "Fatty acids, total monounsaturated",
  "Fatty acids, total polyunsaturated"
)

is_macro_or_fatty_acid_detail <- function(colnm) {
  if (colnm %in% macro_exact) return(TRUE)
  if (grepl("^MUFA |^PUFA |^SFA ", colnm, perl = TRUE)) return(TRUE)
  FALSE
}

pred_names <- nm[!drop_id & !vapply(nm, is_macro_or_fatty_acid_detail, logical(1L))]
X <- dat[, pred_names, drop = FALSE]
if (!all(sapply(X, is.numeric))) {
  stop("All micronutrient predictors must be numeric.")
}
if (ncol(X) < 3L) {
  stop("Too few micronutrient columns after filtering; check column names.")
}

colnames(X) <- make.names(colnames(X), unique = TRUE)
y <- as.numeric(dat[[y_col]])

if (anyNA(y) || any(!is.finite(y))) stop("Energy contains NA or non-finite values.")
if (anyNA(as.matrix(X))) stop("Predictors contain NA; clean or impute before modeling.")

n <- nrow(X)
set.seed(42)
test_frac <- 0.2
test_idx <- sample.int(n, size = floor(test_frac * n))
train_idx <- setdiff(seq_len(n), test_idx)

X_train <- as.matrix(X[train_idx, , drop = FALSE])
X_test <- as.matrix(X[test_idx, , drop = FALSE])
y_train <- y[train_idx]
y_test <- y[test_idx]

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))
mae <- function(actual, pred) mean(abs(actual - pred), na.rm = TRUE)
r_squared <- function(actual, pred) {
  ss_res <- sum((actual - pred)^2, na.rm = TRUE)
  ss_tot <- sum((actual - mean(actual))^2, na.rm = TRUE)
  if (ss_tot < .Machine$double.eps) return(NA_real_)
  1 - ss_res / ss_tot
}

records <- list()
preds <- list()

# perform normalization on the predictors
X_train <- scale(X_train)

# for x_test use the same mean and standard deviation as the training set
X_test <- scale(X_test, center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))

# ----- OLS -----
fit_lm <- lm(y_train ~ ., data = as.data.frame(X_train))
pred_lm <- as.numeric(predict(fit_lm, newdata = as.data.frame(X_test)))
preds[["OLS (micronutrients only)"]] <- pred_lm
records[["OLS (micronutrients only)"]] <- c(
  RMSE = rmse(y_test, pred_lm),
  MAE = mae(y_test, pred_lm),
  R2 = r_squared(y_test, pred_lm)
)

# ----- Ridge & LASSO -----
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, family = "gaussian", standardize = TRUE)
pred_ridge <- as.numeric(predict(cv_ridge, newx = X_test, s = "lambda.min"))
preds[["Ridge (glmnet)"]] <- pred_ridge
records[["Ridge (glmnet)"]] <- c(
  RMSE = rmse(y_test, pred_ridge),
  MAE = mae(y_test, pred_ridge),
  R2 = r_squared(y_test, pred_ridge)
)

cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, family = "gaussian", standardize = TRUE)
pred_lasso <- as.numeric(predict(cv_lasso, newx = X_test, s = "lambda.min"))
preds[["LASSO (glmnet)"]] <- pred_lasso
records[["LASSO (glmnet)"]] <- c(
  RMSE = rmse(y_test, pred_lasso),
  MAE = mae(y_test, pred_lasso),
  R2 = r_squared(y_test, pred_lasso)
)

# ----- Regression tree -----
fit_rp <- rpart::rpart(y_train ~ ., data = cbind(as.data.frame(X_train), y_train = y_train),
                       method = "anova", control = rpart.control(cp = 0.001, minbucket = 5))
pred_rp <- as.numeric(predict(fit_rp, newdata = as.data.frame(X_test)))
preds[["Regression tree (rpart)"]] <- pred_rp
records[["Regression tree (rpart)"]] <- c(RMSE = rmse(y_test, pred_rp), MAE = mae(y_test, pred_rp), R2 = r_squared(y_test, pred_rp))

# ----- Random forest (mtry ~ sqrt(p); no tuneRF for speed) -----
mtry_rf <- max(1L, floor(sqrt(ncol(X_train))))
fit_rf <- randomForest(x = X_train, y = y_train, mtry = mtry_rf, ntree = 400, importance = FALSE)
pred_rf <- as.numeric(predict(fit_rf, X_test))
preds[["Random forest"]] <- pred_rf
records[["Random forest"]] <- c(RMSE = rmse(y_test, pred_rf), MAE = mae(y_test, pred_rf), R2 = r_squared(y_test, pred_rf))

# ----- k-NN regression -----
center <- colMeans(X_train)
scl <- apply(X_train, 2L, sd)
scl[scl == 0 | is.na(scl)] <- 1
scale_train <- function(M) sweep(sweep(M, 2L, center, FUN = "-"), 2L, scl, FUN = "/")
X_tr_s <- scale_train(X_train)
X_te_s <- scale_train(X_test)

k_grid <- seq(3L, min(25L, max(5L, floor(nrow(X_train) / 80))), by = 2L)
fold_id <- sample(rep(1:5, length.out = nrow(X_tr_s)))
cv_rmse <- function(k) {
  err <- numeric(5)
  for (f in 1:5) {
    tr <- fold_id != f
    va <- fold_id == f
    kn <- FNN::knn.reg(train = X_tr_s[tr, , drop = FALSE], test = X_tr_s[va, , drop = FALSE],
                       y = y_train[tr], k = k)
    err[f] <- rmse(y_train[va], kn$pred)
  }
  mean(err)
}
cv_err <- sapply(k_grid, cv_rmse)
best_k <- k_grid[which.min(cv_err)]
kn_final <- FNN::knn.reg(train = X_tr_s, test = X_te_s, y = y_train, k = best_k)
pred_knn <- as.numeric(kn_final$pred)
knn_name <- paste0("k-NN regression (k=", best_k, ")")
preds[[knn_name]] <- pred_knn
records[[knn_name]] <- c(
  RMSE = rmse(y_test, pred_knn),
  MAE = mae(y_test, pred_knn),
  R2 = r_squared(y_test, pred_knn)
)

# ----- Output -----
tab <- as.data.frame(do.call(rbind, records))
tab$Model <- names(records)
tab <- tab[, c("Model", "RMSE", "MAE", "R2")]
tab <- tab[order(tab$RMSE), ]
rownames(tab) <- NULL

# ----- Plots: actual vs predicted (test set) -----
plots_dir <- if (dir.exists("plots")) "plots" else file.path("..", "plots")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

plot_actual_vs_pred_panel <- function(y_act, y_pr, main) {
  lim <- range(c(y_act, y_pr), na.rm = TRUE)
  plot(y_act, y_pr,
       xlab = "Actual Energy (kcal/100 g)",
       ylab = "Predicted Energy (kcal/100 g)",
       main = main,
       pch = 16, col = grDevices::adjustcolor("steelblue", 0.35),
       asp = 1, xlim = lim, ylim = lim, cex.main = 0.95)
  abline(0, 1, col = "firebrick", lwd = 1.5)
  legend(
    "topleft",
    legend = paste0("Test R2 = ", format(round(r_squared(y_act, y_pr), 3), nsmall = 3)),
    bty = "n",
    inset = 0.02,
    cex = 0.82
  )
}

png(file.path(plots_dir, "regression_aim2_actual_vs_predicted_grid.png"),
    width = 1500, height = 1000, res = 120)
par(mfrow = c(2, 3), mar = c(4.2, 4.2, 3.2, 1), mgp = c(2.2, 0.75, 0), oma = c(0, 0, 2.4, 0))
for (nm in tab$Model) {
  plot_actual_vs_pred_panel(y_test, preds[[nm]], main = nm)
}
mtext(
  "Aim 2: Test set — actual vs predicted Energy (micronutrient predictors; red line = perfect agreement)",
  side = 3, outer = TRUE, line = 0.5, cex = 1.05, font = 2
)
dev.off()

best_nm <- tab$Model[1]
png(file.path(plots_dir, "regression_aim2_actual_vs_predicted_best.png"),
    width = 720, height = 680, res = 120)
par(mar = c(4.5, 4.5, 3.5, 1))
plot_actual_vs_pred_panel(
  y_test, preds[[best_nm]],
  main = paste0("Best test RMSE: ", best_nm)
)
mtext(
  paste0("RMSE = ", round(tab$RMSE[1], 2), ", MAE = ", round(tab$MAE[1], 2), " (test set)"),
  side = 3, line = 0.3, cex = 0.95
)
dev.off()

# ----- Text summaries of fitted models -----
ow <- options(width = 110)
sink(out_summaries, split = FALSE)
cat("Aim 2: Model summaries (training data; micronutrient predictors only)\n")
cat("Data:", csv, "\nTrain n:", length(y_train), "\n\n")

cat(strrep("=", 80), "\n")
cat("1. OLS (micronutrients only)\n")
cat(strrep("=", 80), "\n\n")
print(summary(fit_lm))

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("2. Ridge regression (cv.glmnet, alpha = 0)\n")
cat(strrep("=", 80), "\n\n")
print(cv_ridge)
cat("\nlambda.min:", cv_ridge$lambda.min, "  lambda.1se:", cv_ridge$lambda.1se, "\n")
cat("\nCoefficients at lambda.min (intercept + predictors; rounded):\n")
cr <- as.matrix(coef(cv_ridge, s = "lambda.min"))
print(round(cr, 5))

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("3. LASSO (cv.glmnet, alpha = 1)\n")
cat(strrep("=", 80), "\n\n")
print(cv_lasso)
cat("\nlambda.min:", cv_lasso$lambda.min, "  lambda.1se:", cv_lasso$lambda.1se, "\n")
cat("\nNon-zero coefficients at lambda.min:\n")
cl <- as.matrix(coef(cv_lasso, s = "lambda.min"))
cl <- cl[abs(cl[, 1]) > 1e-10, , drop = FALSE]
print(round(cl, 5))
cat("\nNumber of selected predictors (excluding intercept):", max(0L, nrow(cl) - 1L), "\n")

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("4. Regression tree (rpart)\n")
cat(strrep("=", 80), "\n\n")
print(fit_rp)
cat("\nComplexity table (printcp):\n")
printcp(fit_rp)

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("5. Random forest\n")
cat(strrep("=", 80), "\n\n")
print(fit_rf)

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("6. k-NN regression\n")
cat(strrep("=", 80), "\n\n")
cat("No parametric formula; settings from this script:\n")
cat("  Predictors: z-scaled using training mean and SD.\n")
cat("  k (neighbors):", best_k, "(chosen by 5-fold CV RMSE on training rows).\n")
cat("  CV RMSE by k (training):\n")
print(data.frame(k = k_grid, cv_RMSE = cv_err, row.names = NULL))
sink()
options(ow)

sink(out_metrics, split = FALSE)
cat("Aim 2: Predict Energy from micronutrients only (macros & fatty-acid detail excluded)\n\n")
cat(
  "Interpretation: Models use vitamins, minerals, carotenoids, choline, etc., but not\n",
  "protein, fat, carbohydrate, sugars, water, fiber, alcohol, or individual SFA/MUFA/PUFA\n",
  "chains — so R² is expected to be modest versus macro-based models. This isolates how\n",
  "much caloric level is linearly/nonlinearly associated with micronutrient patterns, which\n",
  "supports follow-up work (e.g. ranking foods by micronutrient density = sum of scaled\n",
  "micronutrients per kcal, or studying residuals from these models).\n\n",
  sep = ""
)
cat("Data:", csv, "\n")
cat("Response:", y_col, " | Micronutrient predictors:", ncol(X_train), "\n")
cat("Excluded (examples): protein, fat, carbs, sugars, water, fiber, alcohol, cholesterol,\n")
cat("  total SFA/MUFA/PUFA, and all MUFA/PUFA/SFA detail columns.\n")
cat("Train n:", length(y_train), " Test n:", length(y_test), "\n")
cat("k-NN CV grid:", paste(k_grid, collapse = ", "), " -> best k:", best_k, "\n")
cat("Random forest: mtry =", mtry_rf, ", ntree = 400\n\n")
cat("Test-set performance (lower RMSE/MAE is better):\n\n")
tab_show <- tab
tab_show[, c("RMSE", "MAE", "R2")] <- round(tab[, c("RMSE", "MAE", "R2")], 4)
print(tab_show, row.names = FALSE)
cat("\nPlots (actual vs predicted, test set):\n")
cat("  ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_grid.png"), winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_best.png"), winslash = "/"), "\n", sep = "")
cat("\nModel summaries (fits on training data):\n")
cat("  ", normalizePath(out_summaries, winslash = "/"), "\n", sep = "")
sink()

message("Wrote ", normalizePath(out_metrics, winslash = "/"))
message("Wrote ", normalizePath(out_summaries, winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_grid.png"), winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_best.png"), winslash = "/"))
message("Best test RMSE: ", tab$Model[1], " (", round(tab$RMSE[1], 3), ")")
