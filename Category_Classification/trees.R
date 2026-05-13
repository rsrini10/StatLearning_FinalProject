library(rpart)          # CART
library(rpart.plot)     # CART visualization
library(randomForest)   # random forests
library(caret)          # to do cross-validation with random forests
library(remotes)
## you'll need to run this line if you don't already have the parttree package installed
## remotes::install_github("grantmcdermott/parttree")
library(parttree)
library(ggplot2)
library(MASS)
library(dplyr)
library(ggplot2)
library(dplyr)
library(scales)

# load the data here
df <- read.csv("supervised_table.csv")
df <- df[,4:69]
df <- df %>%
  rename(food_category = wweia_food_category_description)
# df$food_category <- as.factor(df$food_category)

# perform a 70-30 train-test-split
nums <- c(1:5431)
set.seed(123)
idx_train <- sample(nums, 3802)

train_data <- df[idx_train,]
test_data <- df[-idx_train,]



# random forest trees - no hyperparameter tuning
rf_model<-randomForest(factor(food_category) ~ ., data = train_data)
rf_pred<-predict(rf_model,newdata=test_data)

rf_pred<-as.character(predict(rf_model,newdata=test_data))
true_class <- test_data$food_category

num_misclassifications <- sum(rf_pred != true_class)
terror <- num_misclassifications/1629


misclass_table_1 <- data.frame(
  true_class = true_class,
  predicted_class = rf_pred
) %>%
  group_by(true_class) %>%
  summarise(
    n = n(),
    num_misclassified = sum(predicted_class != true_class),
    misclassification_rate = num_misclassified / n
  ) %>%
  arrange(desc(n))

top10_most_frequent_classes <- misclass_table_1[1:10,] %>%
  arrange(desc(misclassification_rate))


top10_most_frequent_classes <- misclass_table_1[1:10, ] %>%
  arrange(desc(misclassification_rate)) %>%
  mutate(
    label = paste0(
      num_misclassified, "/", n,
      " (", round(100 * misclassification_rate, 1), "%)"
    )
  )

ggplot(
  top10_most_frequent_classes,
  aes(
    x = true_class,
    y = misclassification_rate
  )
) +
  
  geom_col(
    width = 0.7,
    fill = "#4C78A8"
  ) +
  
  geom_text(
    aes(label = label),
    hjust = -0.1,
    size = 4.2,
    fontface = "bold"
  ) +
  
  coord_flip() +
  
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.18))
  ) +
  
  labs(
    title = "Misclassification Rate of 10 Most Common Classes",
    subtitle = "Fraction and percentage misclassified for each class",
    x = NULL,
    y = "Misclassification Rate"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18,
      hjust = 0
    ),
    
    plot.subtitle = element_text(
      size = 12,
      color = "gray40"
    ),
    
    axis.text.y = element_text(
      size = 11,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      size = 11
    ),
    
    panel.grid.major.y = element_blank(),
    
    panel.grid.minor = element_blank(),
    
    plot.margin = margin(15, 40, 15, 15)
  )

# most important variables
var_importance <- importance(rf_model)
importance_table <- cbind(Nutrient = rownames(var_importance), var_importance)
importance_table <- as.data.frame(importance_table)
importance_table$MeanDecreaseGini <- as.double(importance_table$MeanDecreaseGini)

top10 <- importance_table %>%
arrange(desc(MeanDecreaseGini)) %>%
slice(1:10)

ggplot(
  top10,
  aes(
    x = reorder(Nutrient, MeanDecreaseGini),
    y = MeanDecreaseGini
  )
) +
  geom_col(
    fill = "#2C7FB8",
    width = 0.7
  ) +
  coord_flip() +
  
  # Add value labels
  geom_text(
    aes(label = round(MeanDecreaseGini, 2)),
    hjust = -0.1,
    size = 4
  ) +
  
  labs(
    title = "Top 10 Most Important Variables",
    subtitle = "Random Forest Variable Importance (Mean Decrease Gini)",
    x = NULL,
    y = "Gini Importance"
  ) +
  
  # Expand axis so labels fit
  expand_limits(y = max(top10$MeanDecreaseGini) * 1.15) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(
      hjust = 0.5,
      size = 18,
      face = "bold"
    ),
    
    plot.subtitle = element_text(
      hjust = 0.5,
      size = 12,
      color = "gray40"
    ),
    
    axis.text.y = element_text(
      size = 12,
      face = "bold"
    ),
    
    axis.text.x = element_text(size = 11),
    
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    
    plot.margin = margin(10, 30, 10, 10)
  )


# refit random forest with just the top 10 variables
all_columns <- top10$Nutrient
all_columns[11] <- "food_category"
train_data <- train_data[,all_columns]
test_data <- test_data[,all_columns]
# random forest trees - no hyperparameter tuning
rf_model_2<-randomForest(factor(food_category) ~ ., data = train_data)

rf_pred_2<-as.character(predict(rf_model_2,newdata=test_data))
true_class <- test_data$food_category

num_misclassifications <- sum(rf_pred_2 != true_class)
terror <- num_misclassifications/1629

misclass_table_2 <- data.frame(
  true_class = true_class,
  predicted_class = rf_pred_2
) %>%
  group_by(true_class) %>%
  summarise(
    n = n(),
    num_misclassified = sum(predicted_class != true_class),
    misclassification_rate = num_misclassified / n
  ) %>%
  arrange(desc(n))

top10_most_frequent_classes <- misclass_table_2[1:10, ] %>%
  arrange(desc(misclassification_rate)) %>%
  mutate(
    label = paste0(
      num_misclassified, "/", n,
      " (", round(100 * misclassification_rate, 1), "%)"
    )
  )


ggplot(
  top10_most_frequent_classes,
  aes(
    x = true_class,
    y = misclassification_rate
  )
) +
  
  geom_col(
    width = 0.7,
    fill = "#4C78A8"
  ) +
  
  geom_text(
    aes(label = label),
    hjust = -0.1,
    size = 4.2,
    fontface = "bold"
  ) +
  
  coord_flip() +
  
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    expand = expansion(mult = c(0, 0.18))
  ) +
  
  labs(
    title = "Misclassification Rate of 10 Most Common Classes",
    subtitle = "Fraction and percentage misclassified for each class",
    x = NULL,
    y = "Misclassification Rate"
  ) +
  
  theme_minimal(base_size = 14) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 18,
      hjust = 0
    ),
    
    plot.subtitle = element_text(
      size = 12,
      color = "gray40"
    ),
    
    axis.text.y = element_text(
      size = 11,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      size = 11
    ),
    
    panel.grid.major.y = element_blank(),
    
    panel.grid.minor = element_blank(),
    
    plot.margin = margin(15, 40, 15, 15)
  )

# random search for mtry
# Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(123)
mtry <- sqrt(ncol(df))
rf_random <- train(food_category~., data=train_data, method="rf", metric="Accuracy", tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)