library(dplyr)
library(tidyr)

food_info <- read.csv("data/food.csv")
nutrient_info <- read.csv("data/nutrient.csv")
food_nutrient <- read.csv("data/food_nutrient.csv")

# now lets do some joins!
join_1 <- left_join(food_nutrient, food_info, by = "fdc_id")
join_2 <- left_join(join_1, nutrient_info, by = c("nutrient_id" = "nutrient_nbr"))

# now lets extract the important columns!
final_table <- join_2[,c("fdc_id","nutrient_id", "amount","description","name","unit_name")]
names(final_table) <- c("fdc_id","nutrient_id", "Nutrient_Conc(per_100_g)","Food_Name","Nutrient_Name","Nutrient_Unit")


final_table <- final_table %>%
  select("Food_Name", "Nutrient_Name", "Nutrient_Unit", "Nutrient_Conc(per_100_g)")


# final table to be used for unsupervised tasks
final_table_wide <- final_table %>%
  select(Food_Name,
         Nutrient_Name,
         Nutrient_Conc.per_100_g = `Nutrient_Conc(per_100_g)`) %>%
  pivot_wider(
    names_from = Nutrient_Name,
    values_from = Nutrient_Conc.per_100_g
  )

write.csv(final_table_wide,'food_nutrient_conc.csv')

# build the supervised table
food_portion <- read.csv("data/food_portion.csv")
survey <- read.csv("data/survey_fndds_food.csv")
food_category <- read.csv("data/wweia_food_category.csv")

join_1 <- left_join(food_info, food_portion, by = "fdc_id")
join_2 <- left_join(join_1,survey, by = "fdc_id")
join_3 <- left_join(join_2, food_category, by = c("wweia_category_number" = "wweia_food_category"))

table(join_3$wweia_food_category_description)

length(unique(join_3$wweia_food_category_description))
supervised_cols <- join_3[,c("description", "wweia_food_category_description")]
df_unique <- unique(supervised_cols)

supervised_table <- left_join(unsupervised_table, df_unique, by = c("Food_Name" = "description"))
write.csv(supervised_table, "data/supervised_table.csv")
