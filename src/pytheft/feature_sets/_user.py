from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from pytheft.feature_sets._base import FeatureSet


class UserFeatures(FeatureSet):
    """Features computed by user-supplied functions, labelled ``"user"``.

    Created from the ``features`` argument of :func:`pytheft.calculate_features`.
    Each function takes a time series as a 1D NumPy array and returns a number.
    """

    name = "user"

    def __init__(self, features: Mapping[str, Callable[[np.ndarray], float]]):
        self.features = features

    def _validate_params(self) -> None:
        if not isinstance(self.features, Mapping) or not self.features:
            raise TypeError(
                "features must be a non-empty dict mapping feature names to functions, "
                "for example {'mean': np.mean, 'sd': np.std}."
            )
        for name, function in self.features.items():
            if not isinstance(name, str):
                raise TypeError(f"Feature names must be strings, got {name!r}.")
            if not callable(function):
                raise TypeError(f"Feature {name!r} must be a function, got {type(function).__name__}.")

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        rows = [[_evaluate(name, function, x) for name, function in self.features.items()] for x in series]
        return pd.DataFrame(rows, columns=list(self.features), dtype=float)


def _evaluate(name: str, function: Callable, x: np.ndarray) -> float:
    value = function(x)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"Feature function {name!r} must return a single number, got {type(value).__name__}."
        ) from None
