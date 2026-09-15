from __future__ import annotations

import os
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pytheft._utils import quiet
from pytheft.feature_sets._base import FeatureSet

#: The feature functions of tsfeatures, in the order tsfeatures computes by default.
_DEFAULT_FEATURES = (
    "acf_features",
    "arch_stat",
    "crossing_points",
    "entropy",
    "flat_spots",
    "heterogeneity",
    "holt_parameters",
    "lumpiness",
    "nonlinearity",
    "pacf_features",
    "stl_features",
    "stability",
    "hw_parameters",
    "unitroot_kpss",
    "unitroot_pp",
    "series_length",
    "hurst",
)
_OTHER_FEATURES = ("count_entropy", "frequency", "guerrero", "intervals", "sparsity", "statistics")

# Importing tsfeatures sets these, and replaces warnings.warn with a function that does nothing.
_THREAD_VARIABLES = ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")


class TSFeatures(FeatureSet):
    """Features from tsfeatures, Nixtla's Python implementation of the R package tsfeatures.

    Parameters
    ----------
    features : list of str, optional
        Names of the tsfeatures functions to compute, such as
        ``["acf_features", "stl_features"]``. By default, the 17 functions
        tsfeatures computes by default are used. ``"count_entropy"``,
        ``"frequency"``, ``"guerrero"``, ``"intervals"``, ``"sparsity"`` and
        ``"statistics"`` are also available.
    freq : int, default=1
        Seasonal period of the time series, such as 12 for monthly data. With
        the default of 1, the series are treated as non-seasonal; with a
        larger value, the seasonal features ``seas_acf1``, ``seas_pacf``,
        ``seasonal_strength``, ``peak`` and ``trough`` are added.
    scale : bool, default=True
        Standardise each series (to mean 0 and standard deviation 1) before
        computing features, as tsfeatures does by default.

    Notes
    -----
    Features that cannot be computed are NaN. With tsfeatures 0.4.5 and
    statsmodels 0.13 or later, this includes the four ``heterogeneity``
    features (``arch_acf``, ``garch_acf``, ``arch_r2`` and ``garch_r2``),
    because tsfeatures uses an autoregressive model class that statsmodels
    has removed. The Holt-Winters parameters are NaN when ``freq`` is 1.

    ``guerrero`` needs more than ``freq`` values; when it is requested,
    shorter series are skipped with a warning.

    References
    ----------
    Garza, F., Challu, C., Olivares, K. G., and Mergenthaler Canseco, M.
    tsfeatures: Calculates various features from time series data.
    https://github.com/Nixtla/tsfeatures

    Hyndman, R. J., Kang, Y., Montero-Manso, P., O'Hara-Wild, M., Talagala,
    T., Wang, E., and Yang, Y. tsfeatures: Time Series Feature Extraction
    (R package). https://pkg.robjhyndman.com/tsfeatures/
    """

    name = "tsfeatures"
    _required_modules = ("tsfeatures",)
    _extra = "tsfeatures"

    def __init__(self, features: Sequence[str] | None = None, freq: int = 1, scale: bool = True):
        self.features = features
        self.freq = freq
        self.scale = scale

    def _validate_params(self) -> None:
        if self.features is not None:
            if isinstance(self.features, str):
                raise TypeError(f"features must be a list of names, got the string {self.features!r}.")
            if len(self.features) == 0:
                raise ValueError("features must name at least one tsfeatures function.")
            known = _DEFAULT_FEATURES + _OTHER_FEATURES
            for name in self.features:
                if name not in known:
                    raise ValueError(f"Unknown tsfeatures function {name!r}; choose from {', '.join(sorted(known))}.")
        if isinstance(self.freq, (bool, np.bool_)) or not isinstance(self.freq, (int, np.integer)) or self.freq < 1:
            raise ValueError(f"freq must be a positive integer, got {self.freq!r}.")
        if not isinstance(self.scale, (bool, np.bool_)):
            raise TypeError(f"scale must be True or False, got {self.scale!r}.")

    def _feature_names(self) -> list[str]:
        return list(_DEFAULT_FEATURES if self.features is None else self.features)

    def _min_length(self) -> int:
        return int(self.freq) + 1 if "guerrero" in self._feature_names() else 1

    def _supported(self, series: list[np.ndarray]) -> np.ndarray:
        min_length = self._min_length()
        return np.array([len(x) >= min_length for x in series])

    def _requirements(self) -> str:
        return f"at least {self._min_length()} values"

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        args = (self._feature_names(), int(self.freq), bool(self.scale), verbose)
        if n_jobs == 1:
            rows = [_transform(x, *args) for x in series]
        else:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs)(delayed(_transform)(x, *args) for x in series)
        return pd.DataFrame(rows, dtype=float)


def _import_tsfeatures():
    """Import tsfeatures, undoing the changes its import makes to the whole Python process."""
    warn = warnings.warn
    environment = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    numpy_errors = np.geterr()
    try:
        import tsfeatures
    finally:
        warnings.warn = warn
        np.seterr(**numpy_errors)
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return tsfeatures


def _transform(x: np.ndarray, feature_names: list[str], freq: int, scale: bool, verbose: bool) -> dict:
    tsfeatures = _import_tsfeatures()

    # Worker processes do not inherit the caller's silenced warnings and logging.
    with quiet(not verbose):
        if scale:
            x = tsfeatures.utils.scalets(x)
        features = {}
        for name in feature_names:
            # Each function gets its own copy, because intervals modifies its input.
            for feature, value in getattr(tsfeatures, name)(x.copy(), freq).items():
                # As in tsfeatures, the first function to return a feature name takes precedence.
                features.setdefault(feature, value)
    return features
