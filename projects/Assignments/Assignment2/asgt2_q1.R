library(faraway)
library(corrplot)
library(leaps)
library(glmnet)
library(pheatmap)

data(seatpos)
dat <- seatpos
dat$hipcenter <- -dat$hipcenter

X <- dat[, 1:8]
y <- dat$hipcenter

# (a) Correlation heatmap with all eight predictors
cor_mat <- cor(X)
round(cor_mat, 6)
corrplot(cor_mat, method = "color")

# Remove HtShoes from both datasets
dat$HtShoes <- NULL
X$HtShoes <- NULL

# (b) Full linear regression with the remaining predictors
full_model <- lm(hipcenter ~ ., data = dat)
summary(full_model)

# (c) Compare all 128 submodels using AIC, BIC and Mallows' Cp
subsets <- c(
  list(character(0)),
  unlist(lapply(seq_len(ncol(X)), function(k) {
    combn(names(X), k, simplify = FALSE)
  }), recursive = FALSE)
)

models <- lapply(subsets, function(vars) {
  formula <- as.formula(
    paste("hipcenter ~", paste(c("1", vars), collapse = " + "))
  )
  lm(formula, data = dat)
})

sigma2 <- summary(full_model)$sigma^2

scores <- data.frame(
  Model = sapply(models, function(m) {
    paste(deparse(formula(m)), collapse = " ")
  }),
  AIC = sapply(models, AIC),
  BIC = sapply(models, BIC),
  Cp = sapply(models, function(m) {
    deviance(m) / sigma2 - nrow(dat) + 2 * m$rank
  })
)

selected <- scores[c(
  which.min(scores$AIC),
  which.min(scores$BIC),
  which.min(scores$Cp)
), ]
rownames(selected) <- c("AIC", "BIC", "Cp")
print(selected)

# (d) Ridge regression with 5-fold cross-validation
set.seed(782)
folds <- sample(rep(1:5, length.out = nrow(dat)))

ridge_cv <- cv.glmnet(
  x = as.matrix(X),
  y = y,
  alpha = 0,
  foldid = folds,
  lambda = exp(seq(log(1e4), log(1e-4), length.out = 200)),
  standardize = TRUE,
  type.measure = "mse"
)

plot(ridge_cv)

# Selected lambda and coefficients in the original units
ridge_cv$lambda.min
coef(ridge_cv, s = "lambda.min")