library(glmnet)

# Load data 
dat <- read.csv("projects/Assignments/Assignment2/data/Assignment1_Question1.csv")

# Extract the 8 covariates and response y = log(psa)
X_raw <- as.matrix(dat[, 2:9])
y <- log(dat$psa)

# Standardize covariates 
X_star <- scale(X_raw, center = TRUE, scale = TRUE)

# Intercept beta_0 is estimated by the sample mean of y
beta_0 <- mean(y)
cat("Estimated intercept (beta_0):", beta_0, "\n")

# Define a grid of 200 lambda values on a logarithmic scale
lambda_grid <- exp(seq(log(1e3), log(1e-4), length.out = 200))

# (a) Fit Ridge regression over the lambda grid

# 5-fold Cross-Validation to choose optimal lambda
set.seed(782)
folds <- sample(rep(1:5, length.out = nrow(X_star)))

ridge_cv <- cv.glmnet(
  x = X_star,
  y = y,
  alpha = 0,
  lambda = lambda_grid,
  foldid = folds,
  standardize = FALSE,
  type.measure = "mse"
)

# Plot CV MSE curve
plot(ridge_cv)

# Optimal lambda and corresponding Ridge coefficients
lambda_opt <- ridge_cv$lambda.min
cat("Optimal lambda (5-fold CV):", lambda_opt, "\n")
print(coef(ridge_cv, s = "lambda.min"))

# Coefficients shrinkage plot with value at optimal hyperparameter
ridge_fit <- glmnet(
  x = X_star,
  y = y,
  alpha = 0,
  lambda = lambda_grid,
  standardize = FALSE
)

# Plot all 8 estimated regression coefficients as a function of lambda in a single panel
matplot(
  x = ridge_fit$lambda,
  y = t(as.matrix(ridge_fit$beta)),
  type = "l",
  log = "x",
  lty = 1,
  lwd = 2,
  col = 1:8,
  xlab = expression(lambda ~ "(log scale)"),
  ylab = expression(hat(beta)[j](lambda)),
  main = "Ridge Coefficient Paths for Prostate Cancer Data"
)
abline(h = 0, lty = 2, col = "gray")
legend(
  "topright",
  legend = colnames(X_star),
  col = 1:8,
  lty = 1,
  lwd = 2,
  cex = 0.8
)
abline(v = ridge_cv$lambda.min, lty = 2, col = "red")


# Fit Lasso regression over the lambda grid (alpha = 1)
lasso_fit <- glmnet(
  x = X_star,
  y = y,
  alpha = 1,
  lambda = lambda_grid,
  standardize = FALSE
)

# Compute BIC for each lambda
n <- nrow(X_star)

y_hat <- predict(lasso_fit, newx = X_star)
rss <- colSums((y - y_hat)^2)
df <- lasso_fit$df # Number of non-zero slope coefficients

# BIC formula:  log(RSS / n) + df * log(n) / n
bic_values <- log(rss / n) + df * log(n) / n

# Optimal lambda minimizing BIC
opt_idx <- which.min(bic_values)
lambda_opt_bic <- lasso_fit$lambda[opt_idx]

cat("Optimal lambda (BIC):", lambda_opt_bic, "\n")
cat("Minimum BIC:", bic_values[opt_idx], "\n")
print(coef(lasso_fit, s = lambda_opt_bic))

# 1. Plot BIC curve as a function of lambda
plot(
  x = lasso_fit$lambda,
  y = bic_values,
  type = "l",
  log = "x",
  lwd = 2,
  col = "darkblue",
  xlab = expression(lambda ~ "(log scale)"),
  ylab = "BIC",
  main = "Lasso Model Selection via BIC"
)
abline(v = lambda_opt_bic, lty = 2, col = "red", lwd = 2)
points(lambda_opt_bic, bic_values[opt_idx], col = "red", pch = 19)

# 2. Plot all 8 estimated Lasso coefficients as a function of lambda
matplot(
  x = lasso_fit$lambda,
  y = t(as.matrix(lasso_fit$beta)),
  type = "l",
  log = "x",
  lty = 1,
  lwd = 2,
  col = 1:8,
  xlab = expression(lambda ~ "(log scale)"),
  ylab = expression(hat(beta)[j](lambda)),
  main = "Lasso Coefficient Paths for Prostate Cancer Data"
)
abline(h = 0, lty = 2, col = "gray")
abline(v = lambda_opt_bic, lty = 2, col = "red", lwd = 2)
legend(
  "topright",
  legend = colnames(X_star),
  col = 1:8,
  lty = 1,
  lwd = 2,
  cex = 0.8
)


