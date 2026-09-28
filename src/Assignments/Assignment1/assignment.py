# %%
# Question 1

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

## Preliminary Data Pre-processing
# Imports
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# %%
data = pd.read_csv("data/Assignment1_Question1.csv")
data.head()

# %%
data.describe()
# %%
data.info()
# %%
X = data.iloc[:, 1:-1]
psa = data.iloc[:, -1].values
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X.columns
# %%
y = np.log(psa)
np.mean(y)
# %%
np.exp(np.min(y))  # Verification - should be 0.65
# %%
np.exp(np.quantile(y, 0.25))  # Verification - should be 5.65
# %%
X_scaled_df = pd.DataFrame(X_scaled, columns=data.columns[1:-1])
X_scaled_df.head()

# %%
X_scaled.mean(axis=0).round(10)  # mean is 0 +- tiny numerical errors
# %%
X_scaled.std(axis=0)  # std is 1

# %%
linear_response_df = X.join(pd.DataFrame(y, columns=["log_psa"]))
linear_response_df.head()
corr_matrix = pd.DataFrame(linear_response_df).corr().abs()

# %%
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, linewidths=0.5
)
plt.title("Correlation Matrix", pad=15)
plt.show()

# %%


def fit_all_models(X=X, y=y):
    results = (
        {}
    )  # We are to populate it with keys {binary_selected, dimension, features_selected, coeffs, intercept, SSR, dimension}
    reg = LinearRegression()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)  # Standardize the features
    n_features = X_scaled.shape[1]

    for k in range(1, n_features + 1):
        for features in combinations(
            range(n_features), k
        ):  # all combinations of features of size k
            X_subset = X_scaled[:, features]
            reg.fit(X_subset, y)
            coeffs = reg.coef_
            intercept = reg.intercept_
            SSR = np.sum((y - reg.predict(X_subset)) ** 2)
            binary_selected = [1 if i in features else 0 for i in range(n_features)]
            results[tuple(binary_selected)] = {
                "dimension": k,
                "col_features_selected": features,
                "features_selected": [X.columns[i] for i in features],
                "coeffs": np.round(coeffs, 4),
                "intercept": np.round(intercept, 4),
                "SSR": np.round(SSR, 4),
            }
    binary_selected = [0] * n_features
    SSR = np.sum((y - np.mean(y)) ** 2)
    results[tuple(binary_selected)] = {
        "dimension": 0,
        "col_features_selected": (),
        "features_selected": [],
        "coeffs": np.empty(0),
        "intercept": round(np.mean(y), 4),
        "SSR": round(SSR, 4),
    }

    return results


# %%
results = fit_all_models(X, y)


# %%
def plot_SSR_vs_dimension(X=X, y=y):
    results = fit_all_models(X, y)
    ssr_by_dimension = {k: {"SSRs": []} for k in range(X.shape[1] + 1)}

    for key, value in results.items():
        ssr_by_dimension[value["dimension"]]["SSRs"].append(value["SSR"])

    dimensions, mean_SSRs, std_SSRs, min_SSRs, max_SSRs = [], [], [], [], []

    for dim, ssr_dict in ssr_by_dimension.items():
        if not ssr_dict["SSRs"]:
            continue
        dimensions.append(dim)
        mean_SSRs.append(np.mean(ssr_dict["SSRs"]))
        std_SSRs.append(np.std(ssr_dict["SSRs"]))
        min_SSRs.append(np.min(ssr_dict["SSRs"]))
        max_SSRs.append(np.max(ssr_dict["SSRs"]))

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        dimensions,
        mean_SSRs,
        yerr=std_SSRs,
        fmt="-o",
        color="blue",
        ecolor="orange",
        elinewidth=2,
        capsize=5,
        label="Mean ± Std",
    )

    plt.plot(dimensions, min_SSRs, color="green", marker="o", label="Min SSR", zorder=3)
    plt.scatter(
        dimensions, max_SSRs, color="red", marker="o", label="Max SSR", zorder=3
    )

    plt.title("SSR vs Dimension of Model")
    plt.xlabel("Dimension of Model (Number of Features)")
    plt.ylabel("Sum of Squared Residuals (SSR)")
    plt.xticks(range(max(dimensions) + 1))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


plot_SSR_vs_dimension()
import numpy as np
import plotly.graph_objects as go


def plot_SSR_vs_dimension(X=X, y=y):
    results = fit_all_models(X, y)
    ssr_by_dimension = {k: {"SSRs": []} for k in range(X.shape[1] + 1)}

    for key, value in results.items():
        ssr_by_dimension[value["dimension"]]["SSRs"].append(value["SSR"])

    dimensions, mean_SSRs, std_SSRs, min_SSRs, max_SSRs = [], [], [], [], []

    for dim, ssr_dict in ssr_by_dimension.items():
        if not ssr_dict["SSRs"]:
            continue
        dimensions.append(dim)
        mean_SSRs.append(np.mean(ssr_dict["SSRs"]))
        std_SSRs.append(np.std(ssr_dict["SSRs"]))
        min_SSRs.append(np.min(ssr_dict["SSRs"]))
        max_SSRs.append(np.max(ssr_dict["SSRs"]))

    fig = go.Figure()

    # Points Min (vert), non connectés
    fig.add_trace(
        go.Scatter(
            x=dimensions,
            y=min_SSRs,
            mode="lines + markers",
            marker=dict(color="green", size=8),
            name="Min SSR",
        )
    )

    # Points Max (rouge), non connectés
    fig.add_trace(
        go.Scatter(
            x=dimensions,
            y=max_SSRs,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Max SSR",
        )
    )

    # Moyenne (bleu) connectée avec barres d'erreur (orange)
    fig.add_trace(
        go.Scatter(
            x=dimensions,
            y=mean_SSRs,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            error_y=dict(
                type="data", array=std_SSRs, color="orange", thickness=2, width=6
            ),
            name="Mean ± Std",
        )
    )

    fig.update_layout(
        title="SSR vs Dimension of Model",
        xaxis_title="Dimension of Model (Number of Features)",
        yaxis_title="Sum of Squared Residuals (SSR)",
        xaxis=dict(
            tickmode="linear", tick0=0, dtick=1
        ),  # Force l'affichage de tous les entiers
        template="plotly_white",
    )

    fig.show()


# %%

import plotly.offline as pyo

pyo.init_notebook_mode(connected=True)
plot_SSR_vs_dimension()

# %%
# The (best) SSR is reduced substantially when adding a dimension, up to $d = 3$. After that, the SSR goes down very slightly, up to $d = 7$. Yielding the full model (compared to using 7 dimensions) barely decreases.
# The results suggest that we could select a 3-features model, which seems to be the best trade off between model parcimony and explaining power.

## 1.2 - Forward, Backward and Stepwise Selection
from sklearn.feature_selection import SequentialFeatureSelector

sfs_forward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select="auto",
    direction="forward",
    cv=5,
    scoring="neg_mean_squared_error",  # MSE
)

sfs_backward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select="auto",
    direction="backward",
    cv=5,
    scoring="neg_mean_squared_error",  # MSE
)


def select_best_model_sfs(X=X, y=y, sfs=sfs_forward):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)  # Standardize the features
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    sfs.fit(X_scaled_df, y)
    selected_features = X.columns[sfs.get_support()].tolist()
    print("Selected features:", selected_features)
    return selected_features


sfs_forward = select_best_model_sfs(X, y, sfs_forward)
# 1. Extract the selected features
X_selected_forward = X_scaled[:, X.columns.isin(sfs_forward)]

# %%

sfs_backward = select_best_model_sfs(X, y, sfs_backward)
X_selected_backward = X_scaled[:, X.columns.isin(sfs_backward)]


# %%
def select_best_model_stepwise(X=X, y=y, cv=5):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled_df = pd.DataFrame(scaler.transform(X), columns=X.columns)
    selected_features = []
    remaining_features = list(X.columns)
    current_score = -np.inf

    while remaining_features:
        addition_scores = {}
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            addition_scores[feature] = cross_val_score(
                LinearRegression(),
                X_scaled_df[candidate_features],
                y,
                cv=cv,
                scoring="neg_mean_squared_error",
            ).mean()

        best_feature = max(addition_scores, key=addition_scores.get)
        best_addition_score = addition_scores[best_feature]
        if best_addition_score <= current_score:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        current_score = best_addition_score

        while len(selected_features) > 1:
            removal_scores = {}
            for feature in selected_features:
                candidate_features = [
                    name for name in selected_features if name != feature
                ]
                removal_scores[feature] = cross_val_score(
                    LinearRegression(),
                    X_scaled_df[candidate_features],
                    y,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                ).mean()

            feature_to_remove = max(removal_scores, key=removal_scores.get)
            best_removal_score = removal_scores[feature_to_remove]
            if best_removal_score <= current_score:
                break

            selected_features.remove(feature_to_remove)
            remaining_features.append(feature_to_remove)
            current_score = best_removal_score

    print("Selected features (stepwise):", selected_features)
    return selected_features


sfs_stepwise = select_best_model_stepwise(X, y)
X_selected_stepwise = X_scaled[:, X.columns.isin(sfs_stepwise)]

# %% FSLR and L2-Boosting
nu = 0.1
iterations = 100


def compute_residuals(y, y_pred):
    return y - y_pred


def fslr(X, y, nu=nu, iterations=iterations, tol=None):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    n_samples, n_features = X_scaled.shape
    coefficients = np.zeros(n_features)
    intercept = np.mean(y)
    y_pred = np.full_like(y, intercept)
    residuals = compute_residuals(y, y_pred)
    ssr_0 = np.sum(residuals**2)
    ssr_history = [ssr_0]

    for _ in range(iterations):
        best_feature_index = None
        best_correlation = 0

        for feature_index in range(
            n_features
        ):  # Select feature most correlated with residuals
            feature_column = X_scaled[:, feature_index]
            correlation = np.abs(np.corrcoef(feature_column, residuals)[0, 1])

            if correlation > best_correlation:
                best_correlation = correlation
                best_feature_index = feature_index

        if best_feature_index is not None:
            feature_column = X_scaled[:, best_feature_index]
            lr = LinearRegression()
            lr.fit(
                feature_column.reshape(-1, 1), residuals
            )  # fit the column with the residuals
            update = nu * np.sign(lr.coef_[0]) * feature_column

            residuals -= update
            new_ssr = np.sum(residuals**2)
            old_ssr = ssr_history[-1]
            if (
                tol is not None and (old_ssr - new_ssr) <= tol
            ):  # if New update doesn't improve SSR by more than tol, stop the iterations
                break
            y_pred += update
            ssr_history.append(np.sum(residuals**2))
            coefficients[best_feature_index] += nu * np.sign(lr.coef_[0])

    return coefficients, intercept, ssr_history


# %%
fslr_coefficients, fslr_intercept, ssr_history = fslr(
    X, y, nu=nu, iterations=iterations
)
selected_features_fslr = [
    X.columns[i] for i in range(len(fslr_coefficients)) if fslr_coefficients[i] != 0
]
print("Selected features (FSLR):", selected_features_fslr)
# %%
print("FSLR Coefficients:", fslr_coefficients)
print("FSLR Intercept:", fslr_intercept)


# %%
def plot_ssr_history(ssr_history, method_name="FSLR"):
    iterations = np.arange(len(ssr_history))
    best_iteration = int(np.argmin(ssr_history))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=ssr_history,
            mode="lines+markers",
            name="SSR",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[best_iteration],
            y=[ssr_history[best_iteration]],
            mode="markers",
            name="Minimum SSR",
            marker=dict(color="firebrick", size=10),
        )
    )

    fig.update_layout(
        title=f"SSR with {method_name}",
        xaxis_title="Iteration",
        yaxis_title="Sum of Squared Residuals (SSR)",
        template="plotly_white",
    )
    fig.show()


# %%

plot_ssr_history(ssr_history, method_name="FSLR")

# The plot suggests that we could stop the iterations, using for example a zero tolerance.abs

# %%

fslr_coefficients, fslr_intercept, ssr_history = fslr(
    X, y, nu=nu, iterations=iterations, tol=0
)
selected_features_fslr = [
    X.columns[i] for i in range(len(fslr_coefficients)) if fslr_coefficients[i] != 0
]
print("Selected features (FSLR):", selected_features_fslr)
print("Number of iterations", len(ssr_history) - 1)

# %%
plot_ssr_history(ssr_history, method_name="FSLR with tol=0")


# %%
def l2_boosting(X, y, nu=nu, iterations=iterations, tol=None):
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    n_samples, n_features = X_scaled.shape
    coefficients = np.zeros(n_features)
    intercept = np.mean(y)
    y_pred = np.full_like(y, intercept)
    residuals = compute_residuals(y, y_pred)
    ssr_0 = np.sum(residuals**2)
    ssr_history = [ssr_0]

    for _ in range(iterations):
        best_feature_index = None
        best_correlation = 0

        for feature_index in range(
            n_features
        ):  # Select feature most correlated with residuals
            feature_column = X_scaled[:, feature_index]
            correlation = np.abs(np.corrcoef(feature_column, residuals)[0, 1])

            if correlation > best_correlation:
                best_correlation = correlation
                best_feature_index = feature_index

        if best_feature_index is not None:
            feature_column = X_scaled[:, best_feature_index]
            lr = LinearRegression()
            lr.fit(
                feature_column.reshape(-1, 1), residuals
            )  # fit the column with the residuals
            update = nu * lr.coef_[0] * feature_column
            y_pred += update
            residuals -= update
            new_ssr = np.sum(residuals**2)
            old_ssr = ssr_history[-1]
            if (
                tol is not None and (old_ssr - new_ssr) <= tol
            ):  # if New update doesn't improve SSR by more than tol, stop the iterations
                break
            ssr_history.append(np.sum(residuals**2))
            coefficients[best_feature_index] += nu * lr.coef_[0]

    return coefficients, intercept, ssr_history


# %%
l2_coefficients, l2_intercept, ssr_history = l2_boosting(
    X, y, nu=nu, iterations=iterations, tol=0
)
selected_features_l2 = [
    X.columns[i] for i in range(len(l2_coefficients)) if l2_coefficients[i] != 0
]
print("Selected features (L2):", selected_features_l2)
print("Number of iterations", len(ssr_history) - 1)

# %%
plot_ssr_history(ssr_history, method_name="L2 Boosting")
# %%
l2_coefficients, l2_intercept, ssr_history = l2_boosting(
    X, y, nu=nu, iterations=iterations, tol=0.1
)
selected_features_l2 = [
    X.columns[i] for i in range(len(l2_coefficients)) if l2_coefficients[i] != 0
]
print("Selected features (L2):", selected_features_l2)
print("Number of iterations", len(ssr_history) - 1)
# %%
plot_ssr_history(ssr_history, method_name="L2 Boosting with tol = 0.1")
# %% Question 2
# Each row is one simulated sample; each column is one observation.
from sklearn.model_selection import KFold

R = 100
n = 50
beta = np.array([1.5, 0.0, 2.0, 0.0, 1.5, 0.0])
CRITERIA = ("Mallows_Cp", "AIC", "BIC", "CV", "GCV", "CV_5", "CV_10")


def generate_response_samples(R, n, filename, random_state=None):
    """Return the full design and an explicit (R, n) array of responses.

    Y_samples[r, i] is observation i in simulated sample r. The supplied
    design includes the intercept column; independent N(0, 1) errors are
    generated for every observation in every sample.
    """
    X_design = pd.read_csv(filename).to_numpy(dtype=float)
    if R < 1 or X_design.shape != (n, len(beta)):
        raise ValueError(
            f"R must be positive and the design must have shape {(n, len(beta))}; "
            f"got R={R}, shape={X_design.shape}."
        )
    if not np.allclose(X_design[:, 0], 1):
        raise ValueError("The first design column must be the intercept (all ones).")

    rng = np.random.default_rng(random_state)
    mean_response = X_design @ beta
    errors = rng.normal(loc=0.0, scale=1.0, size=(R, n))
    Y_samples = mean_response[np.newaxis, :] + errors
    return X_design, Y_samples


# %%
def fit_all_models_criteria(X_design, y, cv=(5, 10)):
    """Score all predictor subsets, always fitting an intercept.

    X_design contains predictors only. Feature tuples use zero-based indices.
    CV is leave-one-out CV, computed exactly from OLS residuals and leverage.
    CV_d uses d fixed, contiguous folds, shared by all candidate models and
    all response samples. Scores are mean squared errors over observations.
    Set cv=None to omit the additional fold-based scores.
    """
    X_design = np.asarray(X_design, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_design.ndim != 2 or y.shape != (X_design.shape[0],):
        raise ValueError(
            "Expected a predictor matrix and one response per observation."
        )
    if not np.isfinite(X_design).all() or not np.isfinite(y).all():
        raise ValueError("Predictors and responses must be finite.")
    n_samples, n_features = X_design.shape
    full_design = np.column_stack((np.ones(n_samples), X_design))
    full_parameters = n_features + 1
    if n_samples <= full_parameters:
        raise ValueError(
            "The full model must have positive residual degrees of freedom."
        )
    if np.linalg.matrix_rank(full_design) != full_parameters:
        raise ValueError("The full design must have linearly independent columns.")
    full_residuals = y - full_design @ np.linalg.lstsq(full_design, y, rcond=None)[0]
    sigma_squared = np.sum(full_residuals**2) / (n_samples - full_parameters)
    if sigma_squared <= 0:
        raise ValueError("The full-model variance estimate must be positive.")

    fold_counts = () if cv is None else ((cv,) if isinstance(cv, int) else tuple(cv))
    folds = {
        d: list(KFold(n_splits=d, shuffle=False).split(X_design)) for d in fold_counts
    }
    model_results = {}
    for k in range(n_features + 1):
        for features in combinations(range(n_features), k):
            # The ones column also makes the empty subset an intercept-only fit.
            design = full_design[:, [0] + [i + 1 for i in features]]
            coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
            residuals = y - design @ coefficients
            ssr = np.sum(residuals**2)
            parameters = k + 1
            q, _ = np.linalg.qr(design, mode="reduced")
            leverage = np.sum(q**2, axis=1)
            result = {
                "dimension": k,
                "features_selected": features,
                "SSR": ssr,
                "Mallows_Cp": ssr / sigma_squared - n_samples + 2 * parameters,
                "AIC": n_samples * np.log(ssr / n_samples) + 2 * parameters,
                "BIC": n_samples * np.log(ssr / n_samples)
                + parameters * np.log(n_samples),
                "CV": np.mean((residuals / (1 - leverage)) ** 2),
                "GCV": (ssr / n_samples) / (1 - parameters / n_samples) ** 2,
            }
            for d, splits in folds.items():
                held_out_ssr = 0.0
                for train, test in splits:
                    fold_design = design[train]
                    if np.linalg.matrix_rank(fold_design) != parameters:
                        raise ValueError("A CV training design is rank deficient.")
                    fold_coefficients = np.linalg.lstsq(
                        fold_design, y[train], rcond=None
                    )[0]
                    held_out_ssr += np.sum(
                        (y[test] - design[test] @ fold_coefficients) ** 2
                    )
                result[f"CV_{d}"] = held_out_ssr / n_samples
            model_results[features] = result
    return model_results


# %%
def select_best_submodels(X_design, Y_samples, criteria=CRITERIA):
    """Return per-sample winners and percentages for every candidate model.

    All criteria are minimized over all 2**p subsets. Exact ties select the
    smaller model, then the first feature tuple. Percentages use all samples
    as the denominator, even when the summary is subsequently filtered.
    """
    X_design = np.asarray(X_design, dtype=float)
    Y_samples = np.asarray(Y_samples, dtype=float)
    if X_design.ndim != 2 or Y_samples.ndim != 2:
        raise ValueError("X_design and Y_samples must both be two-dimensional arrays.")
    if Y_samples.shape[0] == 0 or Y_samples.shape[1] != X_design.shape[0]:
        raise ValueError(
            "Y_samples must have shape (R, number of observations), with R > 0."
        )
    if not criteria or any(criterion not in CRITERIA for criterion in criteria):
        raise ValueError(f"Choose one or more criteria from {CRITERIA}.")

    winners = []
    for sample in Y_samples:
        scores = fit_all_models_criteria(X_design, sample)
        winners.append(
            {
                criterion: min(
                    scores,
                    key=lambda features: (
                        scores[features][criterion],
                        len(features),
                        features,
                    ),
                )
                for criterion in criteria
            }
        )
    selected_models = pd.DataFrame(winners)
    selected_models.index = pd.RangeIndex(1, len(winners) + 1, name="sample")

    rows = []
    for k in range(X_design.shape[1] + 1):
        for features in combinations(range(X_design.shape[1]), k):
            row = {
                "model": "Intercept" + "".join(f" + x{i + 1}" for i in features),
                "features_selected": features,
                "dimension": k,
            }
            for criterion in criteria:
                row[criterion] = (
                    100
                    * sum(winner == features for winner in selected_models[criterion])
                    / len(winners)
                )
            rows.append(row)
    selection_percentages = pd.DataFrame(rows).set_index("model")
    return selected_models, selection_percentages


# %%
# Generate and select independently for each of the three supplied designs.
# The seed makes the simulation reproducible when the cells are rerun.
rng = np.random.default_rng(782)
true_features = tuple(np.flatnonzero(beta[1:]))  # (1, 3), i.e. x2 and x4
simulation_results = {}
selection_tables = {}
for rho, design_name in [(0.0, "I"), (0.5, "II"), (0.75, "III")]:
    full_design, Y_samples = generate_response_samples(
        R, n, f"data/Assignment1_Question2_X_{design_name}.csv", random_state=rng
    )
    X_design = full_design[:, 1:]  # Intercepts are added to every candidate fit.
    selected_models, selection_percentages = select_best_submodels(X_design, Y_samples)
    contains_true_model = selection_percentages["features_selected"].apply(
        lambda features: set(true_features).issubset(features)
    )
    report_table = selection_percentages.loc[contains_true_model, list(CRITERIA)].copy()
    report_table.columns.name = "Selection percentage (%)"
    selection_tables[rho] = report_table
    simulation_results[rho] = {
        "X_design": X_design,
        "Y_samples": Y_samples,
        "selected_models": selected_models,
        "selection_percentages": selection_percentages,
        "report_table": report_table,
    }

# %%
for rho, table in selection_tables.items():
    print(f"Correlation rho = {rho}; {R} samples of {n} observations")
    print(table.to_string())
