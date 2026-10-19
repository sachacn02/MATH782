# MATH 782 — Assignment 2, Question 1 (a–d)
# In R: install.packages(c("faraway", "glmnet"))
# ISLR/ISLR2 contains Hitters (Question 2); seatpos (Question 1) is in faraway.
# Run: Rscript projects/Assignments/Assignment2/asgt2_q1.R [output_directory]
# In RStudio: source("projects/Assignments/Assignment2/asgt2_q1.R")

main <- function(output_dir) {
  required <- c("faraway", "glmnet")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Missing packages: ", paste(missing, collapse = ", "),
         ". Install them with install.packages().")
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Work on a copy to preserve the dataset supplied by faraway.
  dat <- faraway::seatpos
  dat$hipcenter <- -dat$hipcenter
  predictors <- c("Age", "Weight", "HtShoes", "Ht", "Seated", "Arm", "Thigh", "Leg")
  X <- as.matrix(dat[, predictors])
  y <- dat$hipcenter
  n <- nrow(X)
  stopifnot(n == 38L, ncol(X) == 8L, all(is.finite(X)), all(is.finite(y)))

  # (a) Correlations among the eight predictors, excluding the response.
  correlations <- cor(X)
  write.csv(correlations, file.path(output_dir, "q1a_correlations.csv"))
  pdf(file.path(output_dir, "q1a_correlations.pdf"), width = 8, height = 7)
  heatmap(correlations, Rowv = NA, Colv = NA, scale = "none", symm = TRUE,
          col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(101),
          breaks = seq(-1, 1, length.out = 102), margins = c(7, 7),
          main = "Predictor correlations (scale from -1 to 1)")
  dev.off()
  pairs <- which(upper.tri(correlations), arr.ind = TRUE)
  correlation_pairs <- data.frame(
    variable_1 = rownames(correlations)[pairs[, 1]],
    variable_2 = colnames(correlations)[pairs[, 2]],
    correlation = correlations[pairs]
  )
  correlation_pairs <- correlation_pairs[order(-abs(correlation_pairs$correlation)), ]

  # (b) Full model with an intercept; coefficients are in the original units.
  full_model <- lm(hipcenter ~ ., data = dat)
  full_summary <- summary(full_model)
  coefficient_table <- data.frame(term = rownames(coef(full_summary)),
                                  coef(full_summary), row.names = NULL,
                                  check.names = FALSE)
  write.csv(coefficient_table, file.path(output_dir, "q1b_ols_coefficients.csv"),
            row.names = FALSE)
  significant <- rownames(coef(full_summary))[
    rownames(coef(full_summary)) != "(Intercept)" & coef(full_summary)[, 4] < 0.05
  ]

  # (c) Exhaustive search over all 2^8 = 256 submodels, including an intercept.
  # p is the number of regression coefficients, including the intercept.
  # Mallows' Cp always uses the variance estimate from the FULL model.
  # AIC/BIC use logLik(lm), whose degrees of freedom also include sigma².
  sigma2_full <- deviance(full_model) / df.residual(full_model)
  subsets <- c(list(character(0)), unlist(lapply(seq_along(predictors), function(k) {
    combn(predictors, k, simplify = FALSE)
  }), recursive = FALSE))
  models <- lapply(subsets, function(vars) {
    formula <- if (length(vars)) reformulate(vars, response = "hipcenter") else hipcenter ~ 1
    lm(formula, data = dat)
  })
  model_scores <- do.call(rbind, lapply(seq_along(models), function(i) {
    fit <- models[[i]]
    p <- fit$rank
    data.frame(model = paste(deparse(formula(fit)), collapse = " "),
               predictors = paste(subsets[[i]], collapse = ", "),
               p = p, RSS = deviance(fit), AIC = AIC(fit), BIC = BIC(fit),
               Cp = deviance(fit) / sigma2_full - n + 2 * p)
  }))
  stopifnot(nrow(model_scores) == 256L,
            all(vapply(models, nobs, numeric(1)) == n),
            isTRUE(all.equal(tail(model_scores$Cp, 1), as.numeric(full_model$rank))))
  selected <- do.call(rbind, lapply(c("AIC", "BIC", "Cp"), function(criterion) {
    best <- which(abs(model_scores[[criterion]] - min(model_scores[[criterion]])) < 1e-8)
    data.frame(criterion = criterion, model_scores[best, ], row.names = NULL)
  }))
  write.csv(model_scores, file.path(output_dir, "q1c_all_submodels.csv"), row.names = FALSE)
  write.csv(selected, file.path(output_dir, "q1c_selected_models.csv"), row.names = FALSE)

  # (d) Ridge: five balanced folds, shared across the entire lambda grid.
  # glmnet standardizes X during fitting and returns coefficients in the
  # original units. The intercept is not penalized.
  # Objective: RSS/(2*n) + lambda/2 * sum(standardized_beta^2).
  set.seed(782)
  fold_id <- sample(rep(seq_len(5), length.out = n))
  lambda_grid <- exp(seq(log(1e4), log(1e-4), length.out = 200))
  ridge_cv <- glmnet::cv.glmnet(
    x = X, y = y, family = "gaussian", alpha = 0,
    lambda = lambda_grid, foldid = fold_id, nfolds = 5,
    type.measure = "mse", standardize = TRUE, intercept = TRUE
  )
  lambda_hat <- ridge_cv$lambda.min
  ridge_coef <- as.matrix(coef(ridge_cv, s = "lambda.min"))
  ridge_table <- data.frame(term = rownames(ridge_coef),
                            estimate = as.numeric(ridge_coef), row.names = NULL)
  cv_table <- data.frame(lambda = ridge_cv$lambda, mse = ridge_cv$cvm,
                          se = ridge_cv$cvsd)
  stopifnot(length(unique(fold_id)) == 5L, all(is.finite(ridge_coef)),
            ridge_cv$cvm[which.min(abs(ridge_cv$lambda - lambda_hat))] == min(ridge_cv$cvm))
  if (lambda_hat %in% range(lambda_grid)) {
    warning("The CV minimum is at a grid boundary: expand the lambda grid.")
  }
  write.csv(ridge_table, file.path(output_dir, "q1d_ridge_coefficients.csv"), row.names = FALSE)
  write.csv(cv_table, file.path(output_dir, "q1d_ridge_cv.csv"), row.names = FALSE)
  write.csv(data.frame(row = seq_len(n), fold = fold_id),
            file.path(output_dir, "q1d_cv_folds.csv"), row.names = FALSE)
  pdf(file.path(output_dir, "q1d_ridge_cv.pdf"), width = 8, height = 6)
  plot(ridge_cv)
  title("Ridge: 5-fold cross-validation", line = 2.5)
  dev.off()

  # Report the results and their interpretation.
  report <- capture.output({
    cat("QUESTION 1 — seatpos (38 drivers)\n",
        "Response: hipcenter = -faraway::seatpos$hipcenter.\n\n", sep = "")
    cat("(a) Strongest pairwise correlations in absolute value:\n")
    print(head(correlation_pairs, 6), row.names = FALSE)
    cat("Height-related measurements are strongly correlated, making their\n",
        "individual effects difficult to distinguish in the full model.\n\n", sep = "")
    cat("(b) Full linear regression model:\n")
    print(full_summary)
    cat("\nPredictors individually significant at the 5% level: ",
        if (length(significant)) paste(significant, collapse = ", ") else "none", ".\n", sep = "")
    cat("These tests are conditional on the other predictors. Collinearity\n",
        "increases coefficient uncertainty; a nonsignificant individual test\n",
        "does not establish the absence of an association with hipcenter.\n",
        "The F-test above assesses the predictors' joint contribution.\n\n", sep = "")
    cat("(c) Minimum scores across all 256 submodels:\n")
    print(selected[, c("criterion", "model", "p", "AIC", "BIC", "Cp")], row.names = FALSE)
    cat("\nCp = RSS/sigma²_full - n + 2p, with p including the intercept.\n")
    if (length(unique(selected$model)) == 1L) {
      cat("All three criteria select the same submodel here.\n")
    } else {
      cat("The criteria do not all select the same submodel.\n")
    }
    cat("BIC penalizes each parameter by log(38), compared with 2 for AIC,\n",
        "so it favors smaller models more strongly. Correlated measurements\n",
        "can replace each other in submodels with similar fit.\n",
        "These criteria compare models; they do not establish causality.\n\n", sep = "")
    cat("(d) Ridge, seed 782, 5 folds, lambda minimizing the CV MSE:\n")
    cat(sprintf("lambda_hat = %.10g\nlambda_1se (for reference) = %.10g\nMinimum CV MSE = %.6f\n",
                lambda_hat, ridge_cv$lambda.1se, min(ridge_cv$cvm)))
    print(ridge_table, row.names = FALSE)
    cat("These coefficients come from fitting all 38 observations at the\n",
        "selected lambda and are expressed in the original predictor units.\n",
        "Ridge shrinks coefficients to stabilize estimation under collinearity;\n",
        "it does not perform exact variable selection. The selected lambda\n",
        "depends on the CV partition, which is saved for reproducibility.\n", sep = "")
  })
  writeLines(report, file.path(output_dir, "q1_report.txt"))
  cat(report, sep = "\n")
  writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
  invisible(list(data = dat, full_model = full_model, scores = model_scores,
                 selected = selected, ridge_cv = ridge_cv, ridge_coefficients = ridge_table))
}

# Resolve the script path independently of the working directory.
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
source_paths <- Filter(Negate(is.null), lapply(sys.frames(), function(frame) frame$ofile))
script_path <- if (length(source_paths)) {
  tail(source_paths, 1)[[1]]
} else if (length(script_arg)) {
  sub("^--file=", "", script_arg[[1]])
} else {
  stop("Run this file with Rscript or source().")
}
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
args <- if (length(source_paths)) character(0) else commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args)) args[[1]] else file.path(script_dir, "results", "question1")
q1_results <- main(output_dir)
