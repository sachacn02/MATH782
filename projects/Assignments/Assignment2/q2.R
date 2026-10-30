library(ISLR2)
library(corrplot)
library(glmnet)

data(Hitters)
names(Hitters)

## Removing missing data
dim(Hitters)
Hitters <- na.omit(Hitters)
dim(Hitters)

X <- model.matrix(Salary ~ ., Hitters)[, -1]
y <- Hitters$Salary

# (a) Correlation heatmap with all predictors
cor_mat <- cor(X)
round(cor_mat, 6)
corrplot(cor_mat, method = "color", tl.cex = 0.8)

# (b) Full linear regression with all predictors
full_model <- lm(Salary ~ ., data = Hitters)
summary(full_model)

# (d) Ridge regression with 5-fold cross-validation
set.seed(782)
folds <- sample(rep(1:5, length.out = nrow(Hitters)))

ridge_cv <- cv.glmnet(
  x = X,
  y = y,
  alpha = 0,
  foldid = folds,
  lambda = exp(seq(log(1e4), log(1e-2), length.out = 200)),
  standardize = TRUE,
  type.measure = "mse"
)

plot(ridge_cv)

# Selected lambda and coefficients in the original units
ridge_cv$lambda.min
coef(ridge_cv, s = "lambda.min")
