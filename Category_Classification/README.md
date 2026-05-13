# Food Category Classification

## Random Forest Model 1

### Data Split
- 70:30 train-test split

### Model Specification
- Predictors: All nutrient variables (65 total)
- Outcome: Food category (171 unique classes)

### Hyperparameters
- Number of trees: 500
- Variables sampled at each split: 8

### Performance
- Misclassification rate: 0.197

---

## Random Forest Model 2

### Model Specification
- Predictors: 10 most important variables
- Outcome: Food category (171 unique classes)

### Hyperparameters
- Number of trees: 500
- Variables sampled at each split: 3

### Performance
- Misclassification rate: 0.251
