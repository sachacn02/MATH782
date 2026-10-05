"""CV selection paths with Mallows' Cp diagnostics for Assignment 1."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SelectionResult:
    selected_features: tuple[str, ...]
    path: pd.DataFrame
    sigma_squared: float


class _CpScorer:
    def __init__(self, X, y):
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError("X must be a DataFrame with unique column names.")
        self.X = X
        self.y = np.asarray(y, dtype=float)
        values = X.to_numpy(dtype=float)
        n, p = values.shape
        if (
            self.y.shape != (n,)
            or not np.isfinite(values).all()
            or not np.isfinite(self.y).all()
        ):
            raise ValueError("X and y must be finite and have matching rows.")
        if p == 0 or n <= p + 1:
            raise ValueError(
                "The full model needs positive residual degrees of freedom."
            )
        self.names = tuple(X.columns)
        self.n = n
        self.design = np.column_stack((np.ones(n), StandardScaler().fit_transform(X)))
        if np.linalg.matrix_rank(self.design) != p + 1:
            raise ValueError("The full design must have linearly independent columns.")
        self.positions = {name: i + 1 for i, name in enumerate(self.names)}
        self.rss_cache = {}
        self.sigma_squared = self.rss(self.names) / (n - p - 1)
        if self.sigma_squared <= 0:
            raise ValueError("The full-model residual variance must be positive.")

    def ordered(self, features):
        return tuple(name for name in self.names if name in features)

    def rss(self, features):
        key = self.ordered(features)
        if key not in self.rss_cache:
            design = self.design[:, [0] + [self.positions[name] for name in key]]
            coefficients = np.linalg.lstsq(design, self.y, rcond=None)[0]
            residuals = self.y - design @ coefficients
            self.rss_cache[key] = float(residuals @ residuals)
        return self.rss_cache[key]

    def cp(self, features):
        return (
            self.rss(features) / self.sigma_squared - self.n + 2 * (len(features) + 1)
        )

    def row(self, path, move, feature, features, cv_score=np.nan):
        chosen = self.ordered(features)
        path.append(
            {
                "iteration": len(path),
                "move": move,
                "feature": feature,
                "features": chosen,
                "rss": self.rss(chosen),
                "cp": self.cp(chosen),
                "cv_mse": -cv_score,
            }
        )


def _cv_score(scorer, features, folds):
    estimator = make_pipeline(StandardScaler(), LinearRegression())
    return cross_val_score(
        estimator,
        scorer.X[list(features)],
        scorer.y,
        cv=folds,
        scoring="neg_mean_squared_error",
    ).mean()


def sequential_cv_path(X, y, *, direction, target_features=None, folds=5):
    """Reproduce sklearn SFS with a fixed target size; also record Cp per step.

    The candidate at each step maximizes five-fold negative MSE. The default
    target is half the predictors, matching n_features_to_select='auto'.
    """
    scorer = _CpScorer(X, y)
    names = scorer.names
    p = len(names)
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")
    if target_features is None:
        target_features = p // 2
    if not isinstance(target_features, int) or not 0 <= target_features <= p:
        raise ValueError("target_features must be an integer from 0 to p.")

    selected = [] if direction == "forward" else list(names)
    path = []
    scorer.row(path, "start", None, selected)
    while len(selected) != target_features:
        if direction == "forward":
            candidates = [name for name in names if name not in selected]
            options = {name: scorer.ordered(selected + [name]) for name in candidates}
            move = "add"
        else:
            candidates = list(selected)
            options = {
                name: scorer.ordered([item for item in selected if item != name])
                for name in candidates
            }
            move = "remove"
        scores = {
            name: _cv_score(scorer, features, folds)
            for name, features in options.items()
        }
        feature = max(candidates, key=scores.get)
        selected = list(options[feature])
        scorer.row(path, move, feature, selected, scores[feature])

    return SelectionResult(
        scorer.ordered(selected), pd.DataFrame(path), scorer.sigma_squared
    )


def stepwise_cv_path(X, y, *, folds=5):
    """Reproduce the CV-based hybrid search and record Cp after each move."""
    scorer = _CpScorer(X, y)
    names = scorer.names
    selected = []
    current_score = -np.inf
    path = []
    scorer.row(path, "start", None, selected)

    while len(selected) < len(names):
        additions = [name for name in names if name not in selected]
        scores = {
            name: _cv_score(scorer, selected + [name], folds) for name in additions
        }
        feature = max(additions, key=scores.get)
        if scores[feature] <= current_score:
            break
        selected.append(feature)
        current_score = scores[feature]
        scorer.row(path, "add", feature, selected, current_score)

        while len(selected) > 1:
            removals = {
                name: [item for item in selected if item != name] for name in selected
            }
            scores = {
                name: _cv_score(scorer, features, folds)
                for name, features in removals.items()
            }
            feature = max(removals, key=scores.get)
            if scores[feature] <= current_score:
                break
            selected = removals[feature]
            current_score = scores[feature]
            scorer.row(path, "remove", feature, selected, current_score)

    return SelectionResult(
        scorer.ordered(selected), pd.DataFrame(path), scorer.sigma_squared
    )


def plot_cp_paths(results, *, ax=None):
    """Plot Mallows' Cp at each visited model for named selection results."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("Provide at least one selection result.")
    if ax is None:
        fig, (ax, detail_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        fig = ax.figure
        detail_ax = None
    n_paths = len(results)
    for index, (label, result) in enumerate(results.items()):
        path = result.path
        # Slight horizontal offsets reveal points shared by forward and stepwise.
        x = path["iteration"] + 0.08 * (index - (n_paths - 1) / 2)
        for panel in (ax, detail_ax):
            if panel is None:
                continue
            visible = path if panel is ax else path.iloc[1:]
            (line,) = panel.plot(
                x.loc[visible.index], visible["cp"], marker="o", label=label
            )
            panel.scatter(
                x.iloc[-1],
                path.iloc[-1]["cp"],
                s=90,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.5,
            )
    for panel in (ax, detail_ax):
        if panel is None:
            continue
        panel.set_xlabel("Iteration")
        panel.set_xticks(range(max(len(result.path) for result in results.values())))
        panel.grid(alpha=0.3)
    ax.set_ylabel(r"Mallows' $C_p$")
    ax.set_title("Full scale")
    if detail_ax is not None:
        later_cp = [
            value for result in results.values() for value in result.path["cp"].iloc[1:]
        ]
        detail_ax.set_ylim(min(0, min(later_cp) - 1), max(later_cp) * 1.2)
        detail_ax.set_xlim(
            0.6, max(len(result.path) for result in results.values()) - 0.6
        )
        detail_ax.set_title("After the first move")
        detail_ax.legend()
    else:
        ax.legend()
    fig.suptitle(r"Mallows' $C_p$ by iteration (selection uses CV-MSE)")
    fig.tight_layout()
    return fig, ax
