# Exploratory data analysis: food_nutrient_conc.csv and supervised_table.csv
# Run from project root: Rscript eda.R   or   source("eda.R")
# Report is written to results/eda_report.txt

options(width = 120)

results_dir <- "results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
out_report <- file.path(results_dir, "eda_report.txt")

path_food <- "food_nutrient_conc.csv"
path_supervised <- "supervised_table.csv"

stopifnot(file.exists(path_food), file.exists(path_supervised))

food <- read.csv(path_food, check.names = FALSE, stringsAsFactors = FALSE)
sup  <- read.csv(path_supervised, check.names = FALSE, stringsAsFactors = FALSE)

eda_basic <- function(df, label) {
  cat("\n", strrep("=", 72), "\n", sep = "")
  cat(label, "\n")
  cat(strrep("=", 72), "\n\n")

  cat("Dimensions (rows x columns): ", nrow(df), " x ", ncol(df), "\n\n", sep = "")

  cat("Column names (first 15):\n")
  print(head(names(df), 15))
  if (ncol(df) > 15) cat("... and", ncol(df) - 15, "more columns\n")
  cat("\n")

  cat("Column classes:\n")
  print(table(sapply(df, class), useNA = "ifany"))
  cat("\n")

  # Missing values: columns with any NA, sorted by count
  n_miss <- sapply(df, function(x) sum(is.na(x)))
  miss_cols <- sort(n_miss[n_miss > 0], decreasing = TRUE)
  if (length(miss_cols)) {
    cat("Columns with missing values (top 20 by count):\n")
    print(head(miss_cols, 20))
  } else {
    cat("No missing values (NA) in any column.\n")
  }
  cat("\n")

  # Numeric columns: summary (drop exported row-index columns)
  num_idx <- sapply(df, is.numeric)
  nm <- names(df)
  if (any(nm == "")) num_idx[nm == ""] <- FALSE
  if ("X" %in% nm && num_idx["X"]) {
    xv <- df[["X"]]
    if (length(xv) == nrow(df) && !any(is.na(xv)) && min(xv) == 1 && max(xv) == nrow(df))
      num_idx["X"] <- FALSE
  }
  if (any(num_idx)) {
    cat("Summary of numeric columns (min / Q1 / median / mean / Q3 / max):\n")
    X <- as.matrix(df[, num_idx, drop = FALSE])
    qs <- apply(X, 2, quantile, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
    mn <- colMeans(X, na.rm = TRUE)
    tab <- rbind(qs[1:4, , drop = FALSE], mean = mn, max = qs[5, ])
    # Too many columns: show first 12 numeric only in full detail
    k <- min(12, ncol(tab))
    print(round(tab[, seq_len(k), drop = FALSE], 4))
    if (ncol(tab) > k) cat("(", ncol(tab) - k, " additional numeric columns omitted here; see summary(df) locally.)\n", sep = "")
  }

  # Character columns (excluding obvious ID/name if single)
  ch_idx <- sapply(df, function(x) is.character(x) || is.factor(x))
  ch_names <- names(df)[ch_idx]
  for (nm in ch_names) {
    u <- unique(df[[nm]])
    nu <- length(u)
    cat("\nCharacter/factor column: ", nm, " — ", nu, " distinct values\n", sep = "")
    if (nu <= 25) {
      print(sort(table(df[[nm]], useNA = "ifany"), decreasing = TRUE))
    } else {
      tb <- sort(table(df[[nm]], useNA = "ifany"), decreasing = TRUE)
      cat("Top 15 levels:\n")
      print(head(tb, 15))
    }
  }
  cat("\n")
}

sink(file(out_report, open = "wt"), split = TRUE)

eda_basic(food, "food_nutrient_conc.csv")

# supervised_table: same nutrients + label column
eda_basic(sup, "supervised_table.csv")

# Explicit label distribution for supervised learning
if ("wweia_food_category_description" %in% names(sup)) {
  cat(strrep("=", 72), "\n")
  cat("Supervised label: wweia_food_category_description\n")
  cat(strrep("=", 72), "\n\n")
  y <- sup$wweia_food_category_description
  cat("Number of classes:", length(unique(y)), "\n")
  cat("Class counts (sorted):\n")
  print(sort(table(y), decreasing = TRUE))
}

# Quick comparison: same row count and aligned Food_Name?
cat("\n", strrep("=", 72), "\n", sep = "")
cat("Cross-dataset checks\n")
cat(strrep("=", 72), "\n\n")
cat("Row counts — food:", nrow(food), " supervised:", nrow(sup), "\n")
if ("Food_Name" %in% names(food) && "Food_Name" %in% names(sup)) {
  cat("Food_Name sets identical:", identical(sort(unique(food$Food_Name)), sort(unique(sup$Food_Name))), "\n")
  cat("Row order Food_Name identical:", identical(food$Food_Name, sup$Food_Name), "\n")
}
nutr_f <- setdiff(names(food), c("", "Food_Name", "X"))
nutr_s <- setdiff(names(sup), c("", "X", "Food_Name", "wweia_food_category_description"))
cat("Nutrient-like columns in food only (not in supervised): ",
    paste(setdiff(nutr_f, nutr_s), collapse = ", "), "\n", sep = "")
cat("Nutrient-like columns in supervised only (not in food): ",
    paste(setdiff(nutr_s, nutr_f), collapse = ", "), "\n", sep = "")

cat("\nDone.\n")

sink()

message("Wrote ", normalizePath(out_report, winslash = "/"))
