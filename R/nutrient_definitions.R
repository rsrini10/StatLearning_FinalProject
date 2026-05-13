# Central macro vs micronutrient column rules for USDA-style nutrient tables
# (e.g. food_nutrient_conc.csv with read.csv(..., check.names = FALSE)).
#
# From project root: source("R/nutrient_definitions.R")
# From regression/:   source("../R/nutrient_definitions.R")

# Exact column names treated as macronutrients / direct energy carriers (excluded from
# "micronutrient-only" analyses).
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

# TRUE if column should be excluded from micronutrient-only predictor sets:
# macros above, plus individual fatty-acid chain columns (SFA/MUFA/PUFA detail).
is_macro_or_fatty_acid_detail <- function(colnm) {
  if (colnm %in% macro_exact) return(TRUE)
  if (grepl("^MUFA |^PUFA |^SFA ", colnm, perl = TRUE)) return(TRUE)
  FALSE
}

# Names of numeric predictors that are not macros / detailed fatty acids.
# Example: micronutrient_predictor_names(names(dat), y_col = "Energy")
micronutrient_predictor_names <- function(nm, y_col = "Energy", id_cols = c("", "Food_Name")) {
  drop_id <- nm %in% c(id_cols, y_col)
  nm[!drop_id & !vapply(nm, is_macro_or_fatty_acid_detail, logical(1L))]
}
