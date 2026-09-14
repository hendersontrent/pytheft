from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pytheft._utils import quiet
from pytheft.feature_sets._base import FeatureSet

_REMOVED_GROUPS = (
    "cusum_detector",
    "robust_stat_detector",
    "bocp_detector",
    "outlier_detector",
    "trend_detector",
    "seasonalities",
    "time",
)


class Kats(FeatureSet):
    """Features from the time-series features module (``TsFeatures``) of Kats.

    Kats cannot be installed alongside NumPy 2, so pytheft includes an adapted
    copy of its ``tsfeatures`` module, with Kats' MIT licence. By default, the
    40 features that ``TsFeatures`` computes by default are returned, as in
    theft: the ``statistics``, ``stl_features``, ``level_shift_features``,
    ``acfpacf_features``, ``special_ac``, ``holt_params`` and ``hw_params``
    groups.

    Parameters
    ----------
    selected_features : list of str, optional
        Compute only these features or feature groups, as in ``TsFeatures``.
    exclude_features : list of str, optional
        Features or feature groups not to compute.
    nowcasting : bool, default=False
        Also compute the 7 ``nowcasting`` features, which Kats does not
        compute by default.
    window_size, spectral_freq, stl_period, nbins, lag_size, acfpacf_lag, window, n_fast, n_slow : int
        Parameters of the features, passed to ``TsFeatures`` (defaults as in Kats).

    Notes
    -----
    Series need at least ``max(2 * stl_period, 2 * acfpacf_lag + 2)`` values
    (14 by default), because shorter series can make the autocorrelation
    features raise an error; they are skipped with a warning. As in Kats,
    features that cannot be computed are NaN: for example, the Holt-Winters
    parameters are NaN for series with values that are not all positive,
    because the model uses a Box-Cox transformation.

    Kats' feature groups based on its changepoint, outlier, trend and
    seasonality detectors, and its calendar features, are not included.

    References
    ----------
    Jiang, X., Srivastava, S., Chatterjee, S., Yu, Y., Handler, J., Zhang, P.,
    Bopardikar, R., Li, D., Lin, Y., Thakore, U., Brundage, M., Holt, G.,
    Komurlu, C., Nagalla, R., Wang, Z., Sun, H., Gao, P., Cheung, W., Gao, J.,
    Wang, Q., Guerard, M., Kazemi, M., Chen, Y., Zhou, C., Lee, S., Laptev, N.,
    Levendovsky, T., Watson, J., Mu, Y., Bhaskar, A., and Zhou, J. (2022).
    Kats (Version 0.2.0). https://github.com/facebookresearch/Kats
    """

    name = "kats"
    _required_modules = ("statsmodels", "scipy")
    _extra = "kats"

    def __init__(
        self,
        selected_features: Sequence[str] | None = None,
        exclude_features: Sequence[str] | None = None,
        nowcasting: bool = False,
        window_size: int = 20,
        spectral_freq: int = 1,
        stl_period: int = 7,
        nbins: int = 10,
        lag_size: int = 30,
        acfpacf_lag: int = 6,
        window: int = 5,
        n_fast: int = 12,
        n_slow: int = 21,
    ):
        self.selected_features = selected_features
        self.exclude_features = exclude_features
        self.nowcasting = nowcasting
        self.window_size = window_size
        self.spectral_freq = spectral_freq
        self.stl_period = stl_period
        self.nbins = nbins
        self.lag_size = lag_size
        self.acfpacf_lag = acfpacf_lag
        self.window = window
        self.n_fast = n_fast
        self.n_slow = n_slow

    def _validate_params(self) -> None:
        # The feature names are defined in the tsfeatures module, which needs statsmodels and scipy.
        self._check_installed()
        from pytheft._vendor.kats.tsfeatures import _FEATURE_GROUP_MAPPING

        known = set(_FEATURE_GROUP_MAPPING).union(*_FEATURE_GROUP_MAPPING.values())
        for argument in ("selected_features", "exclude_features"):
            names = getattr(self, argument)
            if names is None:
                continue
            if isinstance(names, str):
                raise TypeError(f"{argument} must be a list of names, got the string {names!r}.")
            for name in names:
                if name in _REMOVED_GROUPS:
                    raise ValueError(
                        f"The Kats feature group {name!r} is not available in pytheft, because it needs "
                        "other Kats modules or timestamps."
                    )
                if name not in known:
                    raise ValueError(f"Unknown Kats feature or feature group {name!r} in {argument}.")
        if self.selected_features is not None and len(self.selected_features) == 0:
            raise ValueError("selected_features must name at least one feature or feature group.")

    def _tsfeatures_params(self) -> dict:
        params = {
            "window_size": self.window_size,
            "spectral_freq": self.spectral_freq,
            "stl_period": self.stl_period,
            "nbins": self.nbins,
            "lag_size": self.lag_size,
            "acfpacf_lag": self.acfpacf_lag,
            "window": self.window,
            "n_fast": self.n_fast,
            "n_slow": self.n_slow,
            "selected_features": None if self.selected_features is None else list(self.selected_features),
        }
        switches = {name: False for name in self.exclude_features or ()}
        if self.nowcasting:
            switches["nowcasting"] = True
        return {**params, **switches}

    def _min_length(self) -> int:
        return max(5, 2 * int(self.stl_period), 2 * int(self.acfpacf_lag) + 2)

    def _supported(self, series: list[np.ndarray]) -> np.ndarray:
        min_length = self._min_length()
        return np.array([len(x) >= min_length for x in series])

    def _requirements(self) -> str:
        return f"at least {self._min_length()} values"

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        params = self._tsfeatures_params()
        if n_jobs == 1:
            rows = [_transform(params, x, verbose) for x in series]
        else:
            from joblib import Parallel, delayed

            rows = Parallel(n_jobs=n_jobs)(delayed(_transform)(params, x, verbose) for x in series)
        return pd.DataFrame(rows, dtype=float)


def _transform(params: dict, x: np.ndarray, verbose: bool) -> dict:
    from pytheft._vendor.kats.tsfeatures import TsFeatures

    # Worker processes do not inherit the caller's silenced warnings and logging.
    with quiet(not verbose):
        return TsFeatures(**params).transform(x)
