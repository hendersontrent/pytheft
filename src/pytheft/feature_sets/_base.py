"""Base class that every feature set implements."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class FeatureSet(BaseEstimator, ABC):
    """A set of time-series features computed by one library.

    Use a string such as ``"tsfresh"`` for a feature set's defaults, or an
    instance such as ``TSFresh(settings="efficient")`` to configure it.

    Subclasses set :attr:`name`, store their constructor arguments unchanged
    (the scikit-learn convention, which provides ``repr``, ``get_params`` and
    ``set_params``) and implement :meth:`_calculate`. The input series are
    guaranteed to be non-empty, finite, one-dimensional float arrays.
    """

    #: Label used in the ``feature_set`` column of the output.
    name: ClassVar[str]
    #: Modules that must be importable to compute the feature set.
    _required_modules: ClassVar[tuple[str, ...]] = ()
    #: The pytheft extra that installs the required modules.
    _extra: ClassVar[str | None] = None

    def _check_installed(self) -> None:
        missing = [module for module in self._required_modules if importlib.util.find_spec(module) is None]
        if missing:
            hint = f' Install it with: pip install "pytheft[{self._extra}]"' if self._extra else ""
            raise ImportError(
                f"The {self.name!r} feature set requires {', '.join(missing)}, which is not installed.{hint}"
            )

    def _validate_params(self) -> None:
        """Raise an error for invalid parameters, before any computation starts."""

    def _supported(self, series: list[np.ndarray]) -> np.ndarray:
        """Return a boolean mask of the series this feature set can be computed on."""
        return np.ones(len(series), dtype=bool)

    def _requirements(self) -> str:
        """Describe what the series must satisfy, for messages about skipped series."""
        return ""

    @abstractmethod
    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        """Compute the features of each series.

        Returns a DataFrame with one row per series, in input order, and one
        numeric column per feature.
        """
