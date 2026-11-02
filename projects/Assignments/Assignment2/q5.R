library(glmnet)

# ==========================================
# Question 5(b) - EM algorithm
# ==========================================

# 1. Data Loading
X <- as.matrix(read.table(
  "projects/Assignments/Assignment2/data/Assign2_cX.txt",
  header = FALSE
))

Y_vector <- scan(
  "projects/Assignments/Assignment2/data/Assign2_Y.txt"
)

n <- 200
S <- 10

Y <- matrix(Y_vector, nrow = S, ncol = n, byrow = TRUE)

# ==========================================
# 2. Observed log-likelihood
# ==========================================

calc_loglik <- function(y, X, pi_val, beta1, beta2,
                        sigma1_sq, sigma2_sq) {
  
  log_f1 <- log(pi_val) +
    dnorm(y, as.vector(X %*% beta1),
          sd = sqrt(sigma1_sq), log = TRUE)
  
  log_f2 <- log(1 - pi_val) +
    dnorm(y, as.vector(X %*% beta2),
          sd = sqrt(sigma2_sq), log = TRUE)
  
  max_log <- pmax(log_f1, log_f2)
  
  sum(max_log + log(
    exp(log_f1 - max_log) +
      exp(log_f2 - max_log)
  ))
}


# ==========================================
# 3. EM algorithm
# ==========================================

run_em_mle <- function(y, X, max_iter = 1000, tol = 1e-7) {
  
  n <- length(y)
  p <- ncol(X)
  
  # --- Initialization ---
  
  pi_m <- 0.5
  
  # Random balanced partition
  group <- sample(rep(1:2, length.out = n))
  
  idx1 <- which(group == 1)
  idx2 <- which(group == 2)
  
  # Initial OLS estimates
  fit1 <- lm.fit(X[idx1, , drop = FALSE], y[idx1])
  fit2 <- lm.fit(X[idx2, , drop = FALSE], y[idx2])
  
  if (fit1$rank < p || fit2$rank < p) {
    stop("Rank deficient initial design matrix")
  }
  
  beta1_m <- as.vector(fit1$coefficients)
  beta2_m <- as.vector(fit2$coefficients)
  
  # Initial variance estimates
  sig1_sq <- max(mean(fit1$residuals^2), 1e-6)
  sig2_sq <- max(mean(fit2$residuals^2), 1e-6)
  
  loglik_old <- calc_loglik(
    y, X, pi_m, beta1_m, beta2_m,
    sig1_sq, sig2_sq
  )
  
  converged <- FALSE
  
  # --- EM iterations ---
  
  for (iter in 1:max_iter) {
    
    # E-step (numerically stable)
    
    log_f1 <- log(pi_m) +
      dnorm(y, as.vector(X %*% beta1_m),
            sd = sqrt(sig1_sq), log = TRUE)
    
    log_f2 <- log(1 - pi_m) +
      dnorm(y, as.vector(X %*% beta2_m),
            sd = sqrt(sig2_sq), log = TRUE)
    
    max_log <- pmax(log_f1, log_f2)
    
    log_denom <- max_log +
      log(exp(log_f1 - max_log) +
            exp(log_f2 - max_log))
    
    w1 <- exp(log_f1 - log_denom)
    w2 <- 1 - w1
    
    # M-step
    
    # 1. Mixing probability
    pi_new <- mean(w1)
    pi_new <- min(max(pi_new, 1e-6), 1 - 1e-6)
    
    # 2. Weighted least squares
    beta1_new <- as.vector(
      solve(crossprod(X, X * w1),
            crossprod(X, w1 * y))
    )
    
    beta2_new <- as.vector(
      solve(crossprod(X, X * w2),
            crossprod(X, w2 * y))
    )
    
    # 3. Variance updates
    res1_sq <- (y - as.vector(X %*% beta1_new))^2
    res2_sq <- (y - as.vector(X %*% beta2_new))^2
    
    sig1_sq_new <- sum(w1 * res1_sq) / sum(w1)
    sig2_sq_new <- sum(w2 * res2_sq) / sum(w2)
    
    sig1_sq_new <- max(sig1_sq_new, 1e-6)
    sig2_sq_new <- max(sig2_sq_new, 1e-6)
    
    # Convergence check
    loglik_new <- calc_loglik(
      y, X, pi_new, beta1_new, beta2_new,
      sig1_sq_new, sig2_sq_new
    )
    
    # Update parameters
    pi_m <- pi_new
    beta1_m <- beta1_new
    beta2_m <- beta2_new
    sig1_sq <- sig1_sq_new
    sig2_sq <- sig2_sq_new
    
    if (abs(loglik_new - loglik_old) < tol) {
      converged <- TRUE
      loglik_old <- loglik_new
      break
    }
    
    loglik_old <- loglik_new
  }
  
  # Return estimates
  psi_hat <- c(
    pi = pi_m,
    setNames(beta1_m, paste0("beta1_", 0:5)),
    setNames(beta2_m, paste0("beta2_", 0:5)),
    sigma1_sq = sig1_sq,
    sigma2_sq = sig2_sq
  )
  
  return(list(
    psi_hat = psi_hat,
    loglik = loglik_old,
    iterations = iter,
    converged = converged
  ))
}


# ==========================================
# 4. Multiple random initializations
# ==========================================

run_em_multistart <- function(y, X, n_starts = 10,
                              max_iter = 1000, tol = 1e-7) {
  
  results <- vector("list", n_starts)
  
  for (s in 1:n_starts) {
    
    results[[s]] <- tryCatch(
      run_em_mle(y, X, max_iter, tol),
      error = function(e) NULL
    )
  }
  
  # Keep successful runs
  valid <- which(!sapply(results, is.null))
  
  if (length(valid) == 0) {
    stop("All initializations failed.")
  }
  
  # Select the solution with highest log-likelihood
  logliks <- sapply(results[valid], function(x) x$loglik)
  
  best <- results[[valid[which.max(logliks)] ]]
  
  return(best)
}


# ==========================================
# 5. Run EM on the 10 samples
# ==========================================

set.seed(123)

EM_results <- vector("list", S)

for (s in 1:S) {
  
  EM_results[[s]] <- run_em_multistart(
    y = Y[s, ],
    X = X,
    n_starts = 10
  )
  
  cat("\n============================\n")
  cat("Sample", s, "\n")
  cat("============================\n")
  
  print(EM_results[[s]]$psi_hat)
  
  cat("Log-likelihood:",
      EM_results[[s]]$loglik, "\n")
  
  cat("Iterations:",
      EM_results[[s]]$iterations, "\n")
  
  cat("Converged:",
      EM_results[[s]]$converged, "\n")
}


# ==========================================
# Question 5(c) - Lasso-penalized EM (MPLE) + BIC
# ==========================================

# Matrice des 5 covariables sans la colonne de 1 (glmnet gère l'intercept séparément)
X_cov <- X[, -1, drop = FALSE]

# Optionnel : aligner les composantes (pi <= 0.5) pour éviter le label switching entre les 10 samples
align_components <- function(psi) {
  if (psi["pi"] > 0.5) {
    psi["pi"] <- 1 - psi["pi"]
    b1 <- psi[2:7]; b2 <- psi[8:13]
    psi[2:7]  <- b2
    psi[8:13] <- b1
    s1 <- psi["sigma1_sq"]; s2 <- psi["sigma2_sq"]
    psi["sigma1_sq"] <- s2
    psi["sigma2_sq"] <- s1
  }
  return(psi)
}

# ==========================================
# 1. Penalized EM algorithm for a fixed lambda
# ==========================================

run_em_lasso <- function(y, X, X_cov, lambda_val, init_psi,
                         max_iter = 1000, tol = 1e-6) {
  
  n <- length(y)
  
  # --- Initialization (warm start from MLE of Part b) ---
  init_psi <- align_components(init_psi)
  
  pi_m    <- as.numeric(init_psi["pi"])
  beta1_m <- as.numeric(init_psi[2:7])
  beta2_m <- as.numeric(init_psi[8:13])
  sig1_sq <- as.numeric(init_psi["sigma1_sq"])
  sig2_sq <- as.numeric(init_psi["sigma2_sq"])
  
  loglik_old <- calc_loglik(
    y, X, pi_m, beta1_m, beta2_m,
    sig1_sq, sig2_sq
  )
  
  pen_old <- lambda_val * (sum(abs(beta1_m[-1])) + sum(abs(beta2_m[-1])))
  obj_old <- loglik_old - pen_old
  
  converged <- FALSE
  
  # --- EM iterations ---
  for (iter in 1:max_iter) {
    
    # E-step (numerically stable via log-sum-exp)
    log_f1 <- log(pi_m) +
      dnorm(y, as.vector(X %*% beta1_m),
            sd = sqrt(sig1_sq), log = TRUE)
    
    log_f2 <- log(1 - pi_m) +
      dnorm(y, as.vector(X %*% beta2_m),
            sd = sqrt(sig2_sq), log = TRUE)
    
    max_log <- pmax(log_f1, log_f2)
    
    log_denom <- max_log +
      log(exp(log_f1 - max_log) +
            exp(log_f2 - max_log))
    
    w1 <- exp(log_f1 - log_denom)
    w2 <- 1 - w1
    
    # Guard against degenerate weights
    w1 <- pmin(pmax(w1, 1e-8), 1 - 1e-8)
    w2 <- 1 - w1
    
    # M-step
    
    # 1. Mixing probability
    pi_new <- mean(w1)
    pi_new <- min(max(pi_new, 1e-6), 1 - 1e-6)
    
    # 2. Weighted Lasso updates for beta_1 and beta_2
    if (lambda_val <= 0) {
      beta1_new <- as.vector(
        solve(crossprod(X, X * w1), crossprod(X, w1 * y))
      )
      beta2_new <- as.vector(
        solve(crossprod(X, X * w2), crossprod(X, w2 * y))
      )
    } else {
      # Exact scaling of lambda for glmnet's weighted objective
      lam_glmnet1 <- (lambda_val * sig1_sq) / sum(w1)
      lam_glmnet2 <- (lambda_val * sig2_sq) / sum(w2)
      
      fit1 <- glmnet(
        x = X_cov, y = y, weights = w1,
        alpha = 1, lambda = lam_glmnet1,
        standardize = FALSE, intercept = TRUE
      )
      
      fit2 <- glmnet(
        x = X_cov, y = y, weights = w2,
        alpha = 1, lambda = lam_glmnet2,
        standardize = FALSE, intercept = TRUE
      )
      
      beta1_new <- as.vector(coef(fit1, s = lam_glmnet1))
      beta2_new <- as.vector(coef(fit2, s = lam_glmnet2))
    }
    
    # 3. Variance updates
    res1_sq <- (y - as.vector(X %*% beta1_new))^2
    res2_sq <- (y - as.vector(X %*% beta2_new))^2
    
    sig1_sq_new <- max(sum(w1 * res1_sq) / sum(w1), 1e-6)
    sig2_sq_new <- max(sum(w2 * res2_sq) / sum(w2), 1e-6)
    
    # Penalized log-likelihood check
    loglik_new <- calc_loglik(
      y, X, pi_new, beta1_new, beta2_new,
      sig1_sq_new, sig2_sq_new
    )
    
    pen_new <- lambda_val * (sum(abs(beta1_new[-1])) + sum(abs(beta2_new[-1])))
    obj_new <- loglik_new - pen_new
    
    # Update parameters
    pi_m    <- pi_new
    beta1_m <- beta1_new
    beta2_m <- beta2_new
    sig1_sq <- sig1_sq_new
    sig2_sq <- sig2_sq_new
    loglik_old <- loglik_new
    
    if (abs(obj_new - obj_old) < tol) {
      converged <- TRUE
      obj_old <- obj_new
      break
    }
    
    obj_old <- obj_new
  }
  
  # Clean numerical near-zeros in slope coefficients
  beta1_m[-1][abs(beta1_m[-1]) < 1e-4] <- 0
  beta2_m[-1][abs(beta2_m[-1]) < 1e-4] <- 0
  
  # Effective degrees of freedom: active slopes + 5 unpenalized parameters (pi, 2 intercepts, 2 variances)
  df_slopes <- sum(beta1_m[-1] != 0) + sum(beta2_m[-1] != 0)
  df_total  <- df_slopes + 5
  
  # BIC criterion: -2 * l_n(Psi_tilde) + log(n) * df
  bic_val <- -2 * loglik_old + log(n) * df_total
  
  psi_tilde <- c(
    pi = pi_m,
    setNames(beta1_m, paste0("beta1_", 0:5)),
    setNames(beta2_m, paste0("beta2_", 0:5)),
    sigma1_sq = sig1_sq,
    sigma2_sq = sig2_sq
  )
  
  psi_tilde <- align_components(psi_tilde)
  
  return(list(
    psi_tilde  = psi_tilde,
    lambda     = lambda_val,
    bic        = bic_val,
    loglik     = loglik_old,
    obj        = obj_old,
    df         = df_total,
    iterations = iter,
    converged  = converged
  ))
}


# ==========================================
# 2. Grid search over lambda using BIC
# ==========================================

select_lasso_bic <- function(y, X, X_cov, init_psi, lambda_grid,
                             max_iter = 1000, tol = 1e-6) {
  
  n_grid <- length(lambda_grid)
  fits   <- vector("list", n_grid)
  
  # Sequential warm-starting along the lambda grid for stability
  current_init <- init_psi
  
  for (g in seq_len(n_grid)) {
    fits[[g]] <- run_em_lasso(
      y = y, X = X, X_cov = X_cov,
      lambda_val = lambda_grid[g],
      init_psi = current_init,
      max_iter = max_iter, tol = tol
    )
    current_init <- fits[[g]]$psi_tilde
  }
  
  bics <- sapply(fits, function(x) x$bic)
  best_idx <- which.min(bics)
  
  best_fit <- fits[[best_idx]]
  best_fit$bic_path <- data.frame(
    lambda = lambda_grid,
    bic    = bics,
    df     = sapply(fits, function(x) x$df)
  )
  
  return(best_fit)
}


# ==========================================
# 3. Run Lasso + BIC on the 10 samples
# ==========================================

# Grid of tuning parameters lambda
lambda_grid <- seq(0.5, 12, length.out = 40)

MPLE_results <- vector("list", S)

for (s in 1:S) {
  
  MPLE_results[[s]] <- select_lasso_bic(
    y           = Y[s, ],
    X           = X,
    X_cov       = X_cov,
    init_psi    = EM_results[[s]]$psi_hat,
    lambda_grid = lambda_grid
  )
  
  cat("\n============================\n")
  cat("Sample", s, "- LASSO (MPLE)\n")
  cat("============================\n")
  
  cat("Selected lambda (BIC):",
      MPLE_results[[s]]$lambda, "\n")
  
  cat("Minimum BIC:",
      MPLE_results[[s]]$bic, "\n")
  
  cat("Active degrees of freedom:",
      MPLE_results[[s]]$df, "\n\n")
  
  print(round(MPLE_results[[s]]$psi_tilde, 4))
  
  cat("\nLog-likelihood:",
      MPLE_results[[s]]$loglik, "\n")
  
  cat("Iterations:",
      MPLE_results[[s]]$iterations, "\n")
  
  cat("Converged:",
      MPLE_results[[s]]$converged, "\n")
}

# ==========================================
# 4. Summary matrices for the report
# ==========================================

MLE_table  <- t(sapply(1:S, function(s) align_components(EM_results[[s]]$psi_hat)))
MPLE_table <- t(sapply(1:S, function(s) MPLE_results[[s]]$psi_tilde))
rownames(MLE_table) <- rownames(MPLE_table) <- paste0("Sample_", 1:S)

cat("\n=== Summary Table: MLE (Part b) ===\n")
print(round(MLE_table, 4))

cat("\n=== Summary Table: MPLE / LASSO (Part c) ===\n")
print(round(MPLE_table, 4))
