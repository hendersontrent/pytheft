from __future__ import annotations

import numpy as np
import pandas as pd

from pytheft.feature_sets._base import FeatureSet


class Catch22(FeatureSet):
    """The catch22 set of 22 CAnonical Time-series CHaracteristics, computed with pycatch22.

    Parameters
    ----------
    catch24 : bool, default=False
        Also compute the mean and standard deviation (the catch24 set). The
        catch22 features are insensitive to the location and spread of a time
        series, so add these when those properties matter.

    References
    ----------
    Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., and
    Jones, N. S. (2019). catch22: CAnonical Time-series CHaracteristics. *Data
    Mining and Knowledge Discovery*, 33, 1821-1852.
    https://doi.org/10.1007/s10618-019-00647-x
    """

    name = "catch22"
    _required_modules = ("pycatch22",)
    _extra = "catch22"

    def __init__(self, catch24: bool = False):
        self.catch24 = catch24

    def _validate_params(self) -> None:
        if not isinstance(self.catch24, (bool, np.bool_)):
            raise TypeError(f"catch24 must be True or False, got {self.catch24!r}.")

    def _calculate(self, series: list[np.ndarray], *, n_jobs: int, verbose: bool) -> pd.DataFrame:
        import pycatch22

        results = [pycatch22.catch22_all(x.tolist(), catch24=bool(self.catch24)) for x in series]
        return pd.DataFrame([result["values"] for result in results], columns=results[0]["names"])
