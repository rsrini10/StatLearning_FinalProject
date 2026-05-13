# Aim 2: Predict Energy (kcal/100g) from *micronutrient* composition only (regression).
#
# Motivation: Macros (fat, carbohydrate, protein, water, fiber, alcohol) largely
# determine calories (~4/4/9 kcal/g). Excluding them lets the models use vitamins,
# minerals, and related compounds so you can study how well "micronutrient profile"
# aligns with energy — a step toward comparing foods on micronutrient delivery
# relative to calories (e.g. residuals or separate density indices).
#
# Methods: OLS, ridge & LASSO (glmnet::cv.glmnet), rpart, randomForest, FNN::knn.reg.
# Scaling: z-scores use CV training rows only inside each CV fold (no validation leakage).
# Final fits use scale_from_train(full train, test). Ridge/LASSO use raw training x with
# glmnet standardize=TRUE (glmnet standardizes within each CV training subset for lambda).
# Headline metric: test RMSE (MAE also reported; no R²/Adj R² for cross-method comparison).
# Tuning: shared 5-fold CV; rpart (cp x minbucket x minsplit), RF (mtry x ntree), k-NN (k).
# Discriminant analysis does not apply to continuous Energy; omitted.
#
# Run from project root:
#   Rscript regression/predict_calories.R
# Writes metrics and model summaries to results/; PNGs to plots/regression/.
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

nutr_def <- if (file.exists("R/nutrient_definitions.R")) {
  "R/nutrient_definitions.R"
} else if (file.exists(file.path("..", "R", "nutrient_definitions.R"))) {
  file.path("..", "R", "nutrient_definitions.R")
} else {
  stop("Cannot find R/nutrient_definitions.R (run from project root or regression/).")
}
source(nutr_def, local = FALSE)

root <- if (file.exists("food_nutrient_conc.csv")) "." else ".."
results_dir <- file.path(root, "results")
plots_dir <- file.path(root, "plots", "regression")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
out_metrics <- file.path(results_dir, "regression_aim2_metrics.txt")
out_summaries <- file.path(results_dir, "regression_aim2_model_summaries.txt")

dat <- read.csv(csv, check.names = FALSE, stringsAsFactors = FALSE)
y_col <- "Energy"
stopifnot(y_col %in% names(dat))

nm <- names(dat)
# Exclude macronutrients and detailed fatty acids (see R/nutrient_definitions.R).
pred_names <- micronutrient_predictor_names(nm, y_col = y_col)
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

X_train_raw <- as.matrix(X[train_idx, , drop = FALSE])
X_test_raw <- as.matrix(X[test_idx, , drop = FALSE])
y_train <- y[train_idx]
y_test <- y[test_idx]

n_tr <- nrow(X_train_raw)
p_tr <- ncol(X_train_raw)
set.seed(42)
fold_id_train <- sample(rep(1L:5L, length.out = n_tr))

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))
mae <- function(actual, pred) mean(abs(actual - pred), na.rm = TRUE)

# Column z-scores using train_mat only; apply same center/scale to test_mat.
scale_from_train <- function(train_mat, test_mat) {
  center <- colMeans(train_mat)
  scl <- apply(train_mat, 2L, sd)
  scl[!is.finite(scl) | scl == 0] <- 1
  list(
    train = sweep(sweep(train_mat, 2L, center, "-"), 2L, scl, "/"),
    test = sweep(sweep(test_mat, 2L, center, "-"), 2L, scl, "/"),
    center = center,
    scale = scl
  )
}

scaled_final <- scale_from_train(X_train_raw, X_test_raw)
X_train <- scaled_final$train
X_test <- scaled_final$test

records <- list()
preds <- list()

# ----- OLS (no hyperparameters; 5-fold CV RMSE on training for reference only) -----
ols_cv_rmse <- {
  err <- numeric(5L)
  for (f in 1L:5L) {
    tr <- fold_id_train != f
    va <- fold_id_train == f
    sc <- scale_from_train(
      X_train_raw[tr, , drop = FALSE],
      X_train_raw[va, , drop = FALSE]
    )
    df_tr <- cbind(as.data.frame(sc$train), y_train = y_train[tr])
    df_va <- as.data.frame(sc$test)
    fit_f <- lm(y_train ~ ., data = df_tr)
    pred_f <- as.numeric(predict(fit_f, newdata = df_va))
    err[f] <- rmse(y_train[va], pred_f)
  }
  mean(err)
}
fit_lm <- lm(y_train ~ ., data = as.data.frame(X_train))
pred_lm <- as.numeric(predict(fit_lm, newdata = as.data.frame(X_test)))
preds[["OLS (micronutrients only)"]] <- pred_lm
records[["OLS (micronutrients only)"]] <- c(RMSE = rmse(y_test, pred_lm), MAE = mae(y_test, pred_lm))

# ----- Ridge & LASSO: raw training x; glmnet standardize=TRUE (per-CV-fold scaling inside glmnet) -----
cv_ridge <- cv.glmnet(
  X_train_raw,
  y_train,
  alpha = 0,
  family = "gaussian",
  standardize = TRUE,
  nfolds = 5L,
  foldid = fold_id_train
)
pred_ridge <- as.numeric(predict(cv_ridge, newx = X_test_raw, s = "lambda.min"))
preds[["Ridge (glmnet, 5-fold CV)"]] <- pred_ridge
records[["Ridge (glmnet, 5-fold CV)"]] <- c(RMSE = rmse(y_test, pred_ridge), MAE = mae(y_test, pred_ridge))

cv_lasso <- cv.glmnet(
  X_train_raw,
  y_train,
  alpha = 1,
  family = "gaussian",
  standardize = TRUE,
  nfolds = 5L,
  foldid = fold_id_train
)
pred_lasso <- as.numeric(predict(cv_lasso, newx = X_test_raw, s = "lambda.min"))
preds[["LASSO (glmnet, 5-fold CV)"]] <- pred_lasso
records[["LASSO (glmnet, 5-fold CV)"]] <- c(RMSE = rmse(y_test, pred_lasso), MAE = mae(y_test, pred_lasso))

# ----- Regression tree: 5-fold CV grid over cp, minbucket, minsplit -----
cp_grid <- c(1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1)
minbucket_grid <- c(3L, 5L, 10L)
minsplit_grid <- c(10L, 20L, 40L)
rpart_cv_one <- function(cp_val, mb, ms) {
  err <- numeric(5L)
  for (f in 1L:5L) {
    tr <- fold_id_train != f
    va <- fold_id_train == f
    sc <- scale_from_train(
      X_train_raw[tr, , drop = FALSE],
      X_train_raw[va, , drop = FALSE]
    )
    dat_tr <- cbind(as.data.frame(sc$train), y_train = y_train[tr])
    fit_f <- tryCatch(
      rpart::rpart(
        y_train ~ .,
        data = dat_tr,
        method = "anova",
        control = rpart.control(cp = cp_val, minbucket = mb, minsplit = ms)
      ),
      error = function(e) NULL
    )
    if (is.null(fit_f)) {
      err[f] <- Inf
      next
    }
    pred_f <- tryCatch(
      as.numeric(predict(fit_f, newdata = as.data.frame(sc$test))),
      error = function(e) rep(NA_real_, sum(va))
    )
    err[f] <- rmse(y_train[va], pred_f)
  }
  mean(err)
}
rpart_grid <- expand.grid(
  cp = cp_grid,
  minbucket = minbucket_grid,
  minsplit = minsplit_grid,
  stringsAsFactors = FALSE
)
rpart_grid$cv_RMSE <- mapply(rpart_cv_one, rpart_grid$cp, rpart_grid$minbucket, rpart_grid$minsplit)
best_rp_i <- which.min(rpart_grid$cv_RMSE)
best_cp <- rpart_grid$cp[best_rp_i]
best_minbucket <- as.integer(rpart_grid$minbucket[best_rp_i])
best_minsplit <- as.integer(rpart_grid$minsplit[best_rp_i])
fit_rp <- rpart::rpart(
  y_train ~ .,
  data = cbind(as.data.frame(X_train), y_train = y_train),
  method = "anova",
  control = rpart.control(cp = best_cp, minbucket = best_minbucket, minsplit = best_minsplit)
)
pred_rp <- as.numeric(predict(fit_rp, newdata = as.data.frame(X_test)))
preds[["Regression tree (rpart, 5-fold CV)"]] <- pred_rp
records[["Regression tree (rpart, 5-fold CV)"]] <- c(RMSE = rmse(y_test, pred_rp), MAE = mae(y_test, pred_rp))

# ----- Random forest: 5-fold CV grid over mtry and ntree -----
mtry_grid <- sort(unique(as.integer(pmin(p_tr, round(seq(1, p_tr, length.out = min(5L, p_tr)))))))
ntree_grid <- c(200L, 400L, 600L)
rf_cv_one <- function(mtry_val, ntree_val) {
  err <- numeric(5L)
  for (f in 1L:5L) {
    tr <- fold_id_train != f
    va <- fold_id_train == f
    sc <- scale_from_train(
      X_train_raw[tr, , drop = FALSE],
      X_train_raw[va, , drop = FALSE]
    )
    fit_f <- randomForest(
      x = sc$train,
      y = y_train[tr],
      mtry = mtry_val,
      ntree = ntree_val,
      importance = FALSE
    )
    pred_f <- as.numeric(predict(fit_f, sc$test))
    err[f] <- rmse(y_train[va], pred_f)
  }
  mean(err)
}
rf_grid <- expand.grid(mtry = mtry_grid, ntree = ntree_grid, stringsAsFactors = FALSE)
rf_grid$cv_RMSE <- mapply(rf_cv_one, rf_grid$mtry, rf_grid$ntree)
best_rf_i <- which.min(rf_grid$cv_RMSE)
best_mtry_rf <- as.integer(rf_grid$mtry[best_rf_i])
best_ntree_rf <- as.integer(rf_grid$ntree[best_rf_i])
fit_rf <- randomForest(
  x = X_train,
  y = y_train,
  mtry = best_mtry_rf,
  ntree = best_ntree_rf,
  importance = FALSE
)
pred_rf <- as.numeric(predict(fit_rf, X_test))
rf_model_name <- paste0(
  "Random forest (5-fold CV, mtry=",
  best_mtry_rf,
  ", ntree=",
  best_ntree_rf,
  ")"
)
preds[[rf_model_name]] <- pred_rf
records[[rf_model_name]] <- c(RMSE = rmse(y_test, pred_rf), MAE = mae(y_test, pred_rf))

# ----- k-NN regression: 5-fold CV grid over k (per-fold scaling; same folds as above) -----
k_grid <- seq(3L, min(25L, max(5L, floor(n_tr / 80))), by = 2L)
cv_rmse_knn <- function(k) {
  err <- numeric(5L)
  for (f in 1L:5L) {
    tr <- fold_id_train != f
    va <- fold_id_train == f
    sc <- scale_from_train(
      X_train_raw[tr, , drop = FALSE],
      X_train_raw[va, , drop = FALSE]
    )
    kn <- FNN::knn.reg(
      train = sc$train,
      test = sc$test,
      y = y_train[tr],
      k = k
    )
    err[f] <- rmse(y_train[va], kn$pred)
  }
  mean(err)
}
cv_err <- vapply(k_grid, cv_rmse_knn, numeric(1L))
best_k <- k_grid[which.min(cv_err)]
kn_final <- FNN::knn.reg(train = X_train, test = X_test, y = y_train, k = best_k)
pred_knn <- as.numeric(kn_final$pred)
knn_name <- paste0("k-NN regression (k=", best_k, ", 5-fold CV)")
preds[[knn_name]] <- pred_knn
records[[knn_name]] <- c(RMSE = rmse(y_test, pred_knn), MAE = mae(y_test, pred_knn))

# ----- Output -----
tab <- as.data.frame(do.call(rbind, records))
tab$Model <- names(records)
tab <- tab[, c("Model", "RMSE", "MAE")]
tab <- tab[order(tab$RMSE), ]
rownames(tab) <- NULL

# ----- Plots: actual vs predicted (test set) -----
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
    legend = paste0("Test RMSE = ", format(round(rmse(y_act, y_pr), 3), nsmall = 3)),
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
cat("Data:", csv, "\nTrain n:", length(y_train), "\n")
cat("Shared 5-fold CV fold IDs (training rows): set.seed(42); sample(rep(1:5, length.out = n_train)).\n")
cat("Scaling: see header comments in this script (per-fold z-scores for OLS/rpart/RF/k-NN CV; full-train z-scores for their final fits; glmnet on raw x with standardize=TRUE).\n\n")

cat(strrep("=", 80), "\n")
cat("1. OLS (micronutrients only)\n")
cat(strrep("=", 80), "\n\n")
cat("5-fold CV RMSE on training (reference; no tuning):", ols_cv_rmse, "\n\n")
print(summary(fit_lm))

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("2. Ridge regression (cv.glmnet on raw x, alpha = 0, standardize=TRUE, nfolds = 5, foldid)\n")
cat(strrep("=", 80), "\n\n")
cat("Note: Training matrix is unscaled here; glmnet standardizes within each CV training subset.\n")
cat("Test predictions use predict(..., newx = X_test_raw).\n\n")
print(cv_ridge)
cat("\nlambda.min:", cv_ridge$lambda.min, "  lambda.1se:", cv_ridge$lambda.1se, "\n")
cat("\nCoefficients at lambda.min (intercept + predictors; rounded):\n")
cr <- as.matrix(coef(cv_ridge, s = "lambda.min"))
print(round(cr, 5))

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("3. LASSO (cv.glmnet on raw x, alpha = 1, standardize=TRUE, nfolds = 5, foldid)\n")
cat(strrep("=", 80), "\n\n")
cat("Note: Same scaling approach as ridge.\n\n")
print(cv_lasso)
cat("\nlambda.min:", cv_lasso$lambda.min, "  lambda.1se:", cv_lasso$lambda.1se, "\n")
cat("\nNon-zero coefficients at lambda.min:\n")
cl <- as.matrix(coef(cv_lasso, s = "lambda.min"))
cl <- cl[abs(cl[, 1]) > 1e-10, , drop = FALSE]
print(round(cl, 5))
cat("\nNumber of selected predictors (excluding intercept):", max(0L, nrow(cl) - 1L), "\n")

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("4. Regression tree (rpart): 5-fold CV grid over cp, minbucket, minsplit\n")
cat(strrep("=", 80), "\n\n")
cat("Each CV fold: z-score using training fold only. Grid search (mean CV RMSE); chosen triple refit on full training (z-scored with all training rows):\n")
print(rpart_grid[order(rpart_grid$cv_RMSE), , drop = FALSE])
cat("\nChosen cp:", best_cp, "  minbucket:", best_minbucket, "  minsplit:", best_minsplit, "\n\n")
print(fit_rp)
cat("\nComplexity table (printcp):\n")
printcp(fit_rp)

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("5. Random forest: 5-fold CV grid over mtry and ntree\n")
cat(strrep("=", 80), "\n\n")
cat("Full factorial grid (mean CV RMSE); chosen pair refit on full training:\n")
print(rf_grid[order(rf_grid$cv_RMSE), , drop = FALSE])
cat("\nChosen mtry:", best_mtry_rf, "  ntree:", best_ntree_rf, "\n\n")
print(fit_rf)

cat("\n\n", strrep("=", 80), "\n", sep = "")
cat("6. k-NN regression (5-fold CV on k; same fold IDs)\n")
cat(strrep("=", 80), "\n\n")
cat("Predictors for k-NN: z-scored using full training mean/SD for final fit; CV used per-fold scaling.\n")
cat("Chosen k:", best_k, "\n")
cat("CV RMSE by k (training):\n")
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
cat("Tuning: shared 5-fold CV on training (set.seed(42) before fold_id_train assignment).\n")
cat("  OLS: no hyperparameters; training 5-fold RMSE (reference) =", round(ols_cv_rmse, 4), "\n")
cat("  Ridge/LASSO: glmnet on raw training x, standardize=TRUE, nfolds=5 + foldid; lambda = lambda.min; test x unscaled\n")
cat(
  "  rpart: grid cp x minbucket x minsplit; best cp =",
  best_cp,
  ", minbucket =",
  best_minbucket,
  ", minsplit =",
  best_minsplit,
  "\n",
  sep = ""
)
cat(
  "  Random forest: mtry in {",
  paste(mtry_grid, collapse = ", "),
  "}, ntree in {",
  paste(ntree_grid, collapse = ", "),
  "}; best mtry =",
  best_mtry_rf,
  ", best ntree =",
  best_ntree_rf,
  "\n",
  sep = ""
)
cat("  k-NN: k in {", paste(k_grid, collapse = ", "), "}; best k =", best_k, "\n\n", sep = "")
cat("Scaling: OLS / rpart / RF / k-NN use z-scores from full training for final fit; CV for those models refits z-scores using CV training rows only. glmnet uses raw x with standardize=TRUE.\n")
cat(paste0(
  "Test-set performance (primary metric: RMSE; lower RMSE/MAE is better):\n\n"
))
tab_show <- tab
tab_show[, c("RMSE", "MAE")] <- round(tab[, c("RMSE", "MAE")], 4)
print(tab_show, row.names = FALSE)
cat("\nPlots (actual vs predicted, test set):\n")
cat("  ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_grid.png"), winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_best.png"), winslash = "/"), "\n", sep = "")
cat("\nModel summaries (fits on training data):\n")
cat("  ", normalizePath(out_summaries, winslash = "/"), "\n", sep = "")
sink()

# ----- Supplementary: hyperparameter grids (for report tables) -----
out_supp_sum <- file.path(results_dir, "supplementary_aim2_hyperparameter_summary.csv")
out_supp_rp <- file.path(results_dir, "supplementary_aim2_hyperparameter_rpart_grid.csv")
out_supp_rf <- file.path(results_dir, "supplementary_aim2_hyperparameter_rf_grid.csv")
out_supp_knn <- file.path(results_dir, "supplementary_aim2_hyperparameter_knn_k.csv")
out_supp_txt <- file.path(results_dir, "supplementary_aim2_hyperparameter_grids.txt")

supp_summary <- data.frame(
  method = c(
    "Global train/test split",
    "Global CV fold assignment",
    "OLS",
    "Ridge (glmnet)",
    "LASSO (glmnet)",
    "Regression tree (rpart)",
    "Random forest",
    "k-NN regression (FNN)"
  ),
  hyperparameters_searched = c(
    "(none)",
    "(none)",
    "(none)",
    "lambda (alpha fixed at 0)",
    "lambda (alpha fixed at 1)",
    "cp, minbucket, minsplit",
    "mtry, ntree",
    "k (number of neighbors)"
  ),
  search_space = c(
    paste0("Holdout fraction 0.20; set.seed(42); sample.int for test indices; n_train = ", n_tr, ", n_test = ", length(y_test)),
    paste0("5 folds; set.seed(42); fold_id_train <- sample(rep(1:5, length.out = n_train)); passed to glmnet as foldid"),
    "—",
    "glmnet default penalty (lambda) sequence; family = gaussian; standardize = TRUE",
    "glmnet default penalty (lambda) sequence; family = gaussian; standardize = TRUE",
    paste0(
      "Full factorial grid: cp in {",
      paste(cp_grid, collapse = ", "),
      "}, minbucket in {",
      paste(minbucket_grid, collapse = ", "),
      "}, minsplit in {",
      paste(minsplit_grid, collapse = ", "),
      "} (",
      nrow(rpart_grid),
      " combinations); per-fold z-scores from CV training rows only"
    ),
    paste0(
      "Full factorial: mtry in {",
      paste(mtry_grid, collapse = ", "),
      "}; ntree in {",
      paste(ntree_grid, collapse = ", "),
      "} (",
      nrow(rf_grid),
      " combinations); same ntree in each CV fold and in final fit"
    ),
    paste0(
      "Odd k from ",
      min(k_grid),
      " to ",
      max(k_grid),
      " by 2; upper bound = min(25, max(5, floor(n_train/80)))"
    )
  ),
  n_cv_folds = c(NA_integer_, 5L, 5L, 5L, 5L, 5L, 5L, 5L),
  selection_criterion = c(
    NA_character_,
    NA_character_,
    "N/A (5-fold training RMSE reported only as reference)",
    "Minimum cross-validated deviance (lambda.min)",
    "Minimum cross-validated deviance (lambda.min)",
    "Minimum mean validation RMSE across folds",
    "Minimum mean validation RMSE across folds",
    "Minimum mean validation RMSE across folds"
  ),
  chosen_configuration = c(
    NA_character_,
    NA_character_,
    "N/A",
    paste0("lambda.min = ", format(cv_ridge$lambda.min, digits = 10, scientific = TRUE)),
    paste0("lambda.min = ", format(cv_lasso$lambda.min, digits = 10, scientific = TRUE)),
    paste0("cp = ", best_cp, "; minbucket = ", best_minbucket, "; minsplit = ", best_minsplit),
    paste0("mtry = ", best_mtry_rf, "; ntree = ", best_ntree_rf),
    paste0("k = ", best_k)
  ),
  other_fixed_settings = c(
    NA_character_,
    NA_character_,
    "lm(y ~ .); final z-score from full training; 5-fold CV RMSE uses z-score from CV-train only per fold",
    "nfolds = 5; foldid = fold_id_train; raw X_train_raw; glmnet standardize=TRUE; predict with X_test_raw",
    "nfolds = 5; foldid = fold_id_train; raw X_train_raw; glmnet standardize=TRUE; predict with X_test_raw",
    "method = anova; final z-score from full training; CV grid with per-fold z-scores",
    "importance = FALSE; per-fold z-scores in CV; final fit on full-training z-scores",
    "Final fit on full-training z-scores; CV with per-fold z-scores; zero-variance columns use scale 1"
  ),
  stringsAsFactors = FALSE
)

write.csv(supp_summary, out_supp_sum, row.names = FALSE, fileEncoding = "UTF-8")
write.csv(
  rpart_grid[order(rpart_grid$cv_RMSE), , drop = FALSE],
  out_supp_rp,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  rf_grid[order(rf_grid$cv_RMSE), , drop = FALSE],
  out_supp_rf,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  data.frame(k = as.integer(k_grid), mean_5fold_CV_RMSE = as.numeric(cv_err), stringsAsFactors = FALSE),
  out_supp_knn,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

con <- file(out_supp_txt, open = "wt", encoding = "UTF-8")
writeLines(
  c(
    "Supplementary: Aim 2 hyperparameter search and grids",
    strrep("=", 72),
    "",
    "See regression/predict_calories.R for implementation.",
    "",
    "TABLE 1 — Summary (also: supplementary_aim2_hyperparameter_summary.csv)",
    strrep("-", 72),
    ""
  ),
  con
)
write.table(supp_summary, file = con, sep = "\t", row.names = FALSE, quote = FALSE)
writeLines(
  c(
    "",
    strrep("=", 72),
    "TABLE 2 — rpart cp × minbucket × minsplit grid with mean 5-fold CV RMSE (supplementary_aim2_hyperparameter_rpart_grid.csv)",
    strrep("-", 72),
    ""
  ),
  con
)
write.table(
  rpart_grid[order(rpart_grid$cv_RMSE), , drop = FALSE],
  file = con,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
writeLines(
  c(
    "",
    strrep("=", 72),
    "TABLE 3 — Random forest mtry × ntree grid with mean 5-fold CV RMSE (supplementary_aim2_hyperparameter_rf_grid.csv)",
    strrep("-", 72),
    ""
  ),
  con
)
write.table(
  rf_grid[order(rf_grid$cv_RMSE), , drop = FALSE],
  file = con,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
writeLines(
  c(
    "",
    strrep("=", 72),
    "TABLE 4 — k-NN k vs mean 5-fold CV RMSE (supplementary_aim2_hyperparameter_knn_k.csv)",
    strrep("-", 72),
    ""
  ),
  con
)
write.table(
  data.frame(k = as.integer(k_grid), mean_5fold_CV_RMSE = as.numeric(cv_err)),
  file = con,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
close(con)

message("Wrote ", normalizePath(out_supp_sum, winslash = "/"))
message("Wrote ", normalizePath(out_supp_rp, winslash = "/"))
message("Wrote ", normalizePath(out_supp_rf, winslash = "/"))
message("Wrote ", normalizePath(out_supp_knn, winslash = "/"))
message("Wrote ", normalizePath(out_supp_txt, winslash = "/"))
message("Wrote ", normalizePath(out_metrics, winslash = "/"))
message("Wrote ", normalizePath(out_summaries, winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_grid.png"), winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "regression_aim2_actual_vs_predicted_best.png"), winslash = "/"))
message("Best test RMSE: ", tab$Model[1], " (", round(tab$RMSE[1], 3), ")")
