# Exploratory data analysis: food_nutrient_conc.csv and supervised_table.csv
# Run from project root: Rscript eda.R   or   source("eda.R")
# Report is written to results/eda_report.txt

options(width = 120)
library(ggplot2)

# set ggplot2 theme
theme_set(theme_light())

results_dir <- "results"
plots_dir <- file.path("plots", "eda")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
out_report <- file.path(results_dir, "eda_report.txt")

path_food <- "food_nutrient_conc.csv"
path_supervised <- "supervised_table.csv"

stopifnot(file.exists(path_food), file.exists(path_supervised))

food <- read.csv(path_food, check.names = FALSE, stringsAsFactors = FALSE)
sup  <- read.csv(path_supervised, check.names = FALSE, stringsAsFactors = FALSE)

nutr_def <- if (file.exists("R/nutrient_definitions.R")) {
  "R/nutrient_definitions.R"
} else if (file.exists(file.path("..", "R", "nutrient_definitions.R"))) {
  file.path("..", "R", "nutrient_definitions.R")
} else {
  stop("Cannot find R/nutrient_definitions.R (run from project root or parent directory).")
}
source(nutr_def, local = FALSE)

# Table 1: characteristics of food_nutrient_conc.csv
table1_path_csv <- file.path(results_dir, "table1_dataset_characteristics.csv")
table1_path_txt <- file.path(results_dir, "table1_dataset_characteristics.txt")

food_missing <- sum(is.na(food))
food_missing_cols <- sum(sapply(food, function(x) any(is.na(x))))
food_n_cells <- nrow(food) * ncol(food)
food_name_unique <- if ("Food_Name" %in% names(food)) length(unique(food$Food_Name)) else NA_integer_
label_n <- if ("wweia_food_category_description" %in% names(sup)) {
  length(unique(sup$wweia_food_category_description))
} else {
  NA_integer_
}

nm_food <- names(food)
num_idx <- sapply(food, is.numeric)
if (any(nm_food == "")) num_idx[nm_food == ""] <- FALSE

# "Variable distribution" summary: across numeric variables in food_nutrient_conc.csv
num_mat <- as.matrix(food[, num_idx, drop = FALSE])
var_means <- colMeans(num_mat, na.rm = TRUE)
var_sds <- apply(num_mat, 2, sd, na.rm = TRUE)
var_zero_pct <- colMeans(num_mat == 0, na.rm = TRUE) * 100

table1_key <- data.frame(
  Metric = c(
    "Dataset",
    "Number of foods (rows)",
    "Unique food names",
    "Number of columns",
    "Numeric columns (excluding blank index)",
    "Character/factor columns",
    "Number of labels/classes (from supervised_table.csv)",
    "Total missing cells",
    "Columns with missing values",
    "Percent missing cells"
  ),
  Value = c(
    "food_nutrient_conc.csv",
    nrow(food),
    food_name_unique,
    ncol(food),
    sum(num_idx),
    sum(sapply(food, function(x) is.character(x) || is.factor(x))),
    label_n,
    food_missing,
    food_missing_cols,
    sprintf("%.4f%%", 100 * food_missing / food_n_cells)
  ),
  stringsAsFactors = FALSE
)

table1_dist <- data.frame(
  Metric = c(
    "Across numeric variables: median of variable means",
    "Across numeric variables: IQR of variable means",
    "Across numeric variables: median of variable SDs",
    "Across numeric variables: median % zeros",
    "Across numeric variables: max % zeros",
    "Across numeric variables: min % zeros",
    "Energy (kcal/100g): mean",
    "Energy (kcal/100g): median",
    "Energy (kcal/100g): SD",
    "Energy (kcal/100g): min",
    "Energy (kcal/100g): max"
  ),
  Value = c(
    round(median(var_means), 4),
    round(IQR(var_means), 4),
    round(median(var_sds), 4),
    round(median(var_zero_pct), 2),
    round(max(var_zero_pct), 2),
    round(min(var_zero_pct), 2),
    round(mean(food$Energy, na.rm = TRUE), 4),
    round(median(food$Energy, na.rm = TRUE), 4),
    round(sd(food$Energy, na.rm = TRUE), 4),
    round(min(food$Energy, na.rm = TRUE), 4),
    round(max(food$Energy, na.rm = TRUE), 4)
  ),
  stringsAsFactors = FALSE
)

table1 <- rbind(
  data.frame(Section = "Key characteristics", table1_key, stringsAsFactors = FALSE),
  data.frame(Section = "Variable distribution summary", table1_dist, stringsAsFactors = FALSE)
)
write.csv(table1, table1_path_csv, row.names = FALSE)

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

cat(strrep("=", 72), "\n")
cat("Table 1: food_nutrient_conc.csv characteristics\n")
cat(strrep("=", 72), "\n\n")
cat("[Key characteristics]\n")
print(subset(table1, Section == "Key characteristics", select = c(Metric, Value)), row.names = FALSE)
cat("\n[Variable distribution summary]\n")
print(subset(table1, Section == "Variable distribution summary", select = c(Metric, Value)), row.names = FALSE)
cat("\n")
write.table(table1, file = table1_path_txt, sep = "\t", row.names = FALSE, quote = FALSE)

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


# Visualize the Calorie distribution of the foods
cat("\n", strrep("=", 72), "\n", sep = "")
cat("Calorie distribution of the foods\n")
cat(strrep("=", 72), "\n\n")
p_energy <- ggplot(food, aes(x = Energy)) +
  geom_histogram(binwidth = 50, fill = "#4C78A8", color = "white", linewidth = 0.2) +
  labs(title = "Calorie distribution of the foods", x = "Energy (kcal/100g)", y = "Frequency")
ggsave(file.path(plots_dir, "calorie_distribution.png"), p_energy, width = 7, height = 7)

cat("Average Calorie per food: ", mean(food$Energy), "\n")
cat("Median Calorie per food: ", median(food$Energy), "\n")
cat("Max Calorie per food: ", max(food$Energy), "\n")
cat("Min Calorie per food: ", min(food$Energy), "\n")
cat("Standard Deviation of Calorie per food: ", sd(food$Energy), "\n")

# Visualize micronutrient component distributions (micronutrients only)
cat("\n", strrep("=", 72), "\n", sep = "")
cat("Micronutrient component distributions\n")
cat(strrep("=", 72), "\n\n")

micronutr_cols <- micronutrient_predictor_names(names(food), y_col = "Energy")

micro_df <- food[, micronutr_cols, drop = FALSE]
long_micro <- stack(micro_df)
colnames(long_micro) <- c("value", "component")
long_micro_pos <- subset(long_micro, value > 0)
long_micro_pos$log1p_value <- log1p(long_micro_pos$value)

p_micro_facet <- ggplot(long_micro_pos, aes(x = long_micro_pos$log1p_value)) +
  geom_histogram(bins = 30, fill = "#4C78A8", color = "white", linewidth = 0.2) +
  facet_wrap(~ component, scales = "free_x", ncol = 6) +
  labs(
    title = "Micronutrient distributions across foods (positive values only, log1p scale)",
    subtitle = "Zero-valued foods are excluded in each panel; one panel per micronutrient component",
    x = "log(1 + value)",
    y = "Count of foods with value > 0"
  ) +
  theme(base_size = 9) +
  theme(
    strip.text = element_text(size = 7),
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6)
  )
ggsave(
  file.path(plots_dir, "micronutrient_distributions_faceted.png"),
  p_micro_facet,
  width = 16,
  height = 12
)

zero_pct <- colMeans(micro_df == 0, na.rm = TRUE) * 100
zero_df <- data.frame(component = names(zero_pct), zero_pct = as.numeric(zero_pct))
zero_df <- zero_df[order(zero_df$zero_pct, decreasing = TRUE), , drop = FALSE]
zero_df$component <- factor(zero_df$component, levels = rev(zero_df$component))

p_zero <- ggplot(zero_df, aes(x = component, y = zero_pct)) +
  geom_col(fill = "#F58518") +
  coord_flip() +
  labs(
    title = "Micronutrient sparsity by component",
    subtitle = "Percent of foods with value = 0",
    x = "Micronutrient component",
    y = "Percent zeros (%)"
  ) +
  theme_bw(base_size = 10)
ggsave(
  file.path(plots_dir, "micronutrient_zero_percentage.png"),
  p_zero,
  width = 9,
  height = 10
)

cat("Number of micronutrient components visualized: ", length(micronutr_cols), "\n", sep = "")
cat("Micronutrient histogram panels exclude values equal to 0.\n")
cat("Rows used in faceted histograms (value > 0): ", nrow(long_micro_pos), "\n", sep = "")
cat("Median % zeros across micronutrient components: ", round(median(zero_df$zero_pct), 2), "%\n", sep = "")
cat("Max % zeros across micronutrient components: ", round(max(zero_df$zero_pct), 2), "%\n", sep = "")
cat("Saved plot: ", file.path(plots_dir, "micronutrient_distributions_faceted.png"), "\n", sep = "")
cat("Saved plot: ", file.path(plots_dir, "micronutrient_zero_percentage.png"), "\n", sep = "")
cat("\nDone.\n")

sink()

message("Wrote ", normalizePath(out_report, winslash = "/"))
message("Wrote ", normalizePath(table1_path_csv, winslash = "/"))
message("Wrote ", normalizePath(table1_path_txt, winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "micronutrient_distributions_faceted.png"), winslash = "/"))
message("Wrote ", normalizePath(file.path(plots_dir, "micronutrient_zero_percentage.png"), winslash = "/"))
