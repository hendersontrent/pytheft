from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pytheft.feature_sets._base import FeatureSet

_MIN_LENGTH = 100


class HCTSA(FeatureSet):
    """Features from highly comparative time-series analysis (hctsa), computed with pyhctsa.

    pyhctsa computes over 5000 features, so this set is slow: expect around a
    second or more per time series, depending on its length.

    Parameters
    ----------
    config_path : str or path-like, optional
        A pyhctsa YAML configuration file selecting a subset of operations.
        By default, pyhctsa's full configuration is used.

    Notes
    -----
    pyhctsa requires at least 100 values and a series that is not constant;
    other series are skipped with a warning. Operations that fail, or return
    complex numbers, are recorded as NaN.

    References
    ----------
    Moore, J. B., and Fulcher, B. D. (2026). pyhctsa: A Python package for
    highly comparative time-series analysis. *Journal of Open Source
    Software*, 11(123), 10581. https://doi.org/10.21105/joss.10581

    Fulcher, B. D., and Jones, N. S. (2017). hctsa: A computational framework
    for automated time-series phenotyping using massive feature extraction.
    *Cell Systems*, 5(5), 527-531. https://doi.org/10.1016/j.cels.2017.10.001
    """

    name = "hctsa"
    _required_modules = ("pyhctsa",)
    _extra = "hctsa"

    def __init__(self, config_path: str | os.PathLike | None = None):
        self.config_path = config_path

    def _validate_params(self) -> None:
        if self.config_path is not None and not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"pyhctsa configuration file not found: {self.config_path}")

    def _supported(self, series: list[np.ndarray]) -> np.ndarray:
        return np.array([len(x) >= _MIN_LENGTH and bool(np.ptp(x) > 0) for x in series])

    def _requirements(self) -> str:
        return f"at least {_MIN_LENGTH} values and not constant"

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        from pyhctsa.calculator import FeatureCalculator

        calculator = FeatureCalculator(config_path=None if self.config_path is None else str(self.config_path))
        if n_jobs == 1:
            features = calculator.extract(series, verbose=verbose)
        else:
            from pyhctsa.distribute import LocalDistributor

            distributor = LocalDistributor(n_workers=n_jobs)
            try:
                features = calculator.extract(series, verbose=verbose, distributor=distributor)
            finally:
                distributor.close()
        return _clean_output(features.reset_index(drop=True))


def _clean_output(features: pd.DataFrame) -> pd.DataFrame:
    """Convert pyhctsa's output to floats.

    pyhctsa returns some values as one-element arrays, failures as
    ``"Error: ..."`` strings, and, when an operation with several outputs
    fails, a placeholder column named after the operation alongside its
    ``operation.output`` columns. Placeholders are dropped; failures become NaN.
    """
    non_numeric = [
        column
        for column, dtype in features.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
    ]
    if not non_numeric:
        return features.astype(float)

    operations_with_outputs = {column.split(".", 1)[0] for column in features.columns if "." in column}
    placeholders = [
        column
        for column in non_numeric
        if column in operations_with_outputs and all(_is_error(v) or _is_missing(v) for v in features[column])
    ]
    features = features.drop(columns=placeholders)
    converted = {
        column: np.array([_to_float(v) for v in features[column]], dtype=float)
        for column in non_numeric
        if column not in placeholders
    }
    return features.assign(**converted).astype(float)


def _is_error(value) -> bool:
    return isinstance(value, str) and value.startswith("Error")


def _is_missing(value) -> bool:
    return value is None or (not isinstance(value, np.ndarray) and pd.isna(value))


def _to_float(value) -> float:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return np.nan
        value = value.item()
    if value is None or isinstance(value, (str, bytes)):
        return np.nan
    if isinstance(value, (complex, np.complexfloating)):
        return float(value.real) if value.imag == 0 else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
