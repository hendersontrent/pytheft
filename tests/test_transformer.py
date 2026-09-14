import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pytheft import FeatureExtractor, FeatureSet, PyTheftWarning


class Stats(FeatureSet):
    name = "stats"

    def __init__(self, sd=True):
        self.sd = sd

    def _calculate(self, series, *, n_jobs, verbose):
        features = pd.DataFrame({"mean": [x.mean() for x in series]})
        if self.sd:
            features["sd"] = [x.std() for x in series]
        return features


class LengthDependent(FeatureSet):
    """Returns an extra feature only when every series is long, as TSFEL can."""

    name = "varies"

    def _calculate(self, series, *, n_jobs, verbose):
        features = pd.DataFrame({"mean": [x.mean() for x in series]})
        if all(len(x) >= 50 for x in series):
            features["long_only"] = [x[:50].mean() for x in series]
        return features


def test_fit_transform_returns_a_feature_matrix(simulated):
    X, _ = simulated
    extractor = FeatureExtractor(Stats())
    matrix = extractor.fit_transform(X)
    assert matrix.shape == (len(X), 2)
    np.testing.assert_allclose(matrix[:, 0], X.mean(axis=1))
    assert extractor.get_feature_names_out().tolist() == ["stats__mean", "stats__sd"]
    np.testing.assert_allclose(extractor.transform(X), matrix)


def test_transform_uses_the_features_found_when_fitting():
    rng = np.random.default_rng(0)
    extractor = FeatureExtractor(LengthDependent()).fit(rng.standard_normal((3, 60)))
    short = extractor.transform(rng.standard_normal((2, 20)))
    assert short.shape == (2, 2)
    assert np.isnan(short[:, 1]).all()

    extractor.fit(rng.standard_normal((3, 20)))
    assert extractor.transform(rng.standard_normal((2, 60))).shape == (2, 1)


def test_invalid_series_give_rows_of_nan(simulated):
    X = simulated[0].copy()
    X[1, 0] = np.nan
    with pytest.warns(PyTheftWarning, match="Features are NaN for 1 of 12"):
        matrix = FeatureExtractor(Stats()).fit_transform(X)
    assert matrix.shape == (12, 2)
    assert np.isnan(matrix[1]).all()
    assert not np.isnan(np.delete(matrix, 1, axis=0)).any()


def test_list_of_unequal_length_series():
    assert FeatureExtractor(Stats()).fit_transform([np.arange(10.0), np.arange(25.0)]).shape == (2, 2)


def test_pandas_output_keeps_the_dataframe_index(simulated):
    X, _ = simulated
    frame = pd.DataFrame(X, index=[f"s{i}" for i in range(len(X))])
    output = FeatureExtractor(Stats()).set_output(transform="pandas").fit_transform(frame)
    assert output.columns.tolist() == ["stats__mean", "stats__sd"]
    assert output.index.tolist() == frame.index.tolist()


def test_dataframe_with_repeated_index(simulated):
    X, _ = simulated
    frame = pd.DataFrame(X, index=np.zeros(len(X), dtype=int))
    output = FeatureExtractor(Stats()).set_output(transform="pandas").fit_transform(frame)
    np.testing.assert_allclose(output["stats__mean"], X.mean(axis=1))


def test_nested_parameters_and_clone():
    extractor = FeatureExtractor(Stats())
    assert extractor.get_params()["feature_set__sd"] is True
    extractor.set_params(feature_set__sd=False)
    cloned = clone(extractor)
    assert cloned.feature_set is not extractor.feature_set
    assert cloned.fit(np.ones((2, 5))).get_feature_names_out().tolist() == ["stats__mean"]


def test_transform_before_fit(simulated):
    with pytest.raises(NotFittedError):
        FeatureExtractor(Stats()).transform(simulated[0])


def test_pipeline_cross_validation_with_catch22(simulated_large):
    pytest.importorskip("pycatch22")
    X, y = simulated_large
    pipeline = make_pipeline(FeatureExtractor("catch22"), StandardScaler(), LogisticRegression(max_iter=1000))
    assert cross_val_score(pipeline, X, y, cv=3).mean() > 0.8
