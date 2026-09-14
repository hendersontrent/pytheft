from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from pytheft.feature_sets._base import FeatureSet


class TSFEL(FeatureSet):
    """Features from TSFEL (Time Series Feature Extraction Library).

    Parameters
    ----------
    domain : str or list of str, optional
        Feature domains to compute: ``"statistical"``, ``"temporal"``,
        ``"spectral"``, ``"fractal"`` or ``"all"``. By default, TSFEL's
        default configuration is used (statistical, temporal and spectral),
        as in theft.
    fs : float, optional
        Sampling frequency of the time series, used by the spectral features.
        By default, TSFEL's configured value (100 Hz) is used.
    config : dict, optional
        A complete TSFEL feature configuration, such as one returned by
        ``tsfel.get_features_by_tag``. Cannot be combined with ``domain``.

    Notes
    -----
    Some spectral features need a minimum number of values (12 with the
    default configuration); shorter series are skipped with a warning. The
    number of features TSFEL returns also depends on series length.

    References
    ----------
    Barandas, M., Folgado, D., Fernandes, L., Santos, S., Abreu, M., Bota, P.,
    Liu, H., Schultz, T., and Gamboa, H. (2020). TSFEL: Time Series Feature
    Extraction Library. *SoftwareX*, 11, 100456.
    https://doi.org/10.1016/j.softx.2020.100456
    """

    name = "tsfel"
    _required_modules = ("tsfel",)
    _extra = "tsfel"

    def __init__(self, domain: str | list[str] | None = None, fs: float | None = None, config: dict | None = None):
        self.domain = domain
        self.fs = fs
        self.config = config

    def _validate_params(self) -> None:
        if self.domain is not None and self.config is not None:
            raise ValueError("Pass either domain or config to TSFEL, not both.")

    def _feature_config(self) -> dict:
        import tsfel

        if self.config is not None:
            # TSFEL writes the sampling frequency into the configuration it is given.
            return copy.deepcopy(self.config)
        return tsfel.get_features_by_domain(self.domain)

    def _min_length(self) -> int:
        # Features taking n_coeff (such as LPCC) raise an error for series shorter than it.
        lengths = [
            int(feature["parameters"]["n_coeff"])
            for domain in self._feature_config().values()
            for feature in domain.values()
            if feature.get("use") == "yes"
            and isinstance(feature.get("parameters"), dict)
            and "n_coeff" in feature["parameters"]
        ]
        return max(lengths, default=1)

    def _supported(self, series: list[np.ndarray]) -> np.ndarray:
        min_length = self._min_length()
        return np.array([len(x) >= min_length for x in series])

    def _requirements(self) -> str:
        return f"at least {self._min_length()} values"

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        import tsfel

        # TSFEL evaluates every feature twice for 1D input; an (n, 1) column gives
        # identical results in half the time. Its column names are prefixed "0_".
        features = tsfel.time_series_features_extractor(
            self._feature_config(),
            [x[:, np.newaxis] for x in series],
            fs=self.fs,
            verbose=int(verbose),
            n_jobs=None if n_jobs == 1 else n_jobs,
        )
        features.columns = [column.removeprefix("0_") for column in features.columns]
        return features
