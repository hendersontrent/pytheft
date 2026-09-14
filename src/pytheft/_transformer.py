"""The scikit-learn interface: :class:`FeatureExtractor`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from pytheft._calculate import compute_feature_sets
from pytheft._input import prepare_collection, to_collection
from pytheft._utils import effective_n_jobs
from pytheft.feature_sets import resolve_feature_sets


class FeatureExtractor(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer that turns time series into a feature matrix.

    Following scikit-learn conventions, each row of ``X`` is one time series:
    ``X`` can be a 2D array or DataFrame of shape ``(n_series, n_timepoints)``,
    a 3D array of shape ``(n_series, 1, n_timepoints)``, or a list of 1D arrays
    of any length. Use :func:`pytheft.calculate_features` for long-format data.

    Parameters
    ----------
    feature_set : str, FeatureSet or list of these, default="catch22"
        Feature sets to compute, as in :func:`pytheft.calculate_features`.
    features : dict, optional
        Your own features, as a dict mapping feature names to functions.
    z_score : bool, default=False
        Standardise each series before computing features.
    n_jobs : int, optional
        Number of processes for feature sets that support parallelism.
    verbose : bool, default=False
        Print progress and show the feature libraries' own output and warnings.

    Attributes
    ----------
    feature_names_out_ : ndarray of str
        Feature names, formatted ``"{feature_set}__{feature}"``, fixed when
        fitting. Transformed data always has these columns, in this order,
        even if a library returns a different set of features for new data.

    Notes
    -----
    Fitting only records the feature names; nothing is learned from the data.
    Series with missing values, or that a feature set cannot process, give
    rows of NaN so the output stays aligned with ``y``.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipeline = make_pipeline(FeatureExtractor("catch22"), StandardScaler(), LogisticRegression())
    """

    def __init__(self, feature_set="catch22", *, features=None, z_score=False, n_jobs=None, verbose=False):
        self.feature_set = feature_set
        self.features = features
        self.z_score = z_score
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute features on ``X`` to determine the output feature names."""
        self._fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform ``X``, computing its features only once."""
        return self._fit(X)

    def transform(self, X):
        """Compute features for each time series in ``X``.

        Returns
        -------
        ndarray of shape (n_series, n_features)
        """
        check_is_fitted(self, "feature_names_out_")
        return self._compute(X).reindex(columns=self.feature_names_out_).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names, formatted ``"{feature_set}__{feature}"``."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _fit(self, X) -> np.ndarray:
        features = self._compute(X)
        self.feature_names_out_ = np.asarray(features.columns, dtype=object)
        return features.to_numpy(dtype=float)

    def _compute(self, X) -> pd.DataFrame:
        feature_sets = resolve_feature_sets(self.feature_set, self.features)
        n_jobs = effective_n_jobs(self.n_jobs)
        collection = to_collection(X, dataframe_layout="wide")
        prepared, kept = prepare_collection(collection, z_score=self.z_score, on_invalid="nan")
        results = compute_feature_sets(prepared, feature_sets, n_jobs=n_jobs, verbose=self.verbose)

        blocks = []
        for feature_set, computed in results:
            block = computed.set_axis(kept[computed.index.to_numpy()], axis=0)
            block = block.reindex(np.arange(len(collection)))
            block.columns = [f"{feature_set.name}__{name}" for name in block.columns]
            blocks.append(block)
        return pd.concat(blocks, axis=1)
