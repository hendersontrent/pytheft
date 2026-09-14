from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from pytheft.feature_sets._base import FeatureSet

_SETTINGS = {
    "comprehensive": "ComprehensiveFCParameters",
    "efficient": "EfficientFCParameters",
    "minimal": "MinimalFCParameters",
}


class TSFresh(FeatureSet):
    """Features from tsfresh (Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests).

    Parameters
    ----------
    settings : {"comprehensive", "efficient", "minimal"} or dict, default="comprehensive"
        Which features to compute. The names match tsfresh's
        ``ComprehensiveFCParameters`` (all features, as in theft),
        ``EfficientFCParameters`` (without the most expensive features) and
        ``MinimalFCParameters``. A dict is passed to tsfresh as
        ``default_fc_parameters``.

    Notes
    -----
    tsfresh's ``extract_relevant_features`` is deliberately not offered: it
    selects features using class labels, which must happen inside
    cross-validation to avoid leaking information from test data.

    References
    ----------
    Christ, M., Braun, N., Neuffer, J., and Kempa-Liehr, A. W. (2018). Time
    Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh --
    A Python package). *Neurocomputing*, 307, 72-77.
    https://doi.org/10.1016/j.neucom.2018.03.067
    """

    name = "tsfresh"
    _required_modules = ("tsfresh",)
    _extra = "tsfresh"

    def __init__(self, settings: str | Mapping = "comprehensive"):
        self.settings = settings

    def _validate_params(self) -> None:
        if isinstance(self.settings, str):
            if self.settings.lower() not in _SETTINGS:
                raise ValueError(
                    f"Unknown tsfresh settings {self.settings!r}; choose from {', '.join(_SETTINGS)} or pass a dict."
                )
        elif not isinstance(self.settings, Mapping):
            raise TypeError(
                f"settings must be one of {', '.join(_SETTINGS)} or a dict, got {type(self.settings).__name__}."
            )

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import settings as tsfresh_settings

        if isinstance(self.settings, str):
            parameters = getattr(tsfresh_settings, _SETTINGS[self.settings.lower()])()
        else:
            parameters = dict(self.settings)

        lengths = [len(x) for x in series]
        long = pd.DataFrame(
            {
                "id": np.repeat(np.arange(len(series)), lengths),
                "time": np.concatenate([np.arange(length) for length in lengths]),
                "value": np.concatenate(series),
            }
        )
        features = extract_features(
            long,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=parameters,
            n_jobs=0 if n_jobs == 1 else n_jobs,
            disable_progressbar=not verbose,
            show_warnings=verbose,
        )
        # tsfresh sorts its output by id, so restore the input order.
        features = features.reindex(np.arange(len(series))).reset_index(drop=True)
        features.columns = [column.removeprefix("value__") for column in features.columns]
        return features
