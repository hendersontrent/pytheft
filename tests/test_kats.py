"""Checks of the Kats feature set and pytheft's copy of Kats' tsfeatures module.

Kats cannot be installed alongside NumPy 2, so its output is compared with
reference values saved from the original Kats (see tests/data/kats_reference.json).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pytheft import Kats, PyTheftWarning, calculate_features, to_wide

pytest.importorskip("statsmodels")

from pytheft._vendor.kats.tsfeatures import TsFeatures

REFERENCE = json.loads((Path(__file__).parent / "data" / "kats_reference.json").read_text())

# Parameters found by numerical optimisation vary slightly between statsmodels versions.
OPTIMISED = ("holt_", "hw_")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("configuration", "kwargs"), [("default", {}), ("with_nowcasting", {"nowcasting": True})])
def test_tsfeatures_matches_original_kats(configuration, kwargs):
    for name, expected in REFERENCE["features"][configuration].items():
        result = TsFeatures(**kwargs).transform(np.array(REFERENCE["series"][name]))
        assert list(result) == list(expected), name
        for feature, value in expected.items():
            value = np.nan if value is None else value
            tolerance = {"rtol": 1e-3, "atol": 1e-6} if feature.startswith(OPTIMISED) else {"rtol": 1e-7, "atol": 1e-12}
            np.testing.assert_allclose(result[feature], value, equal_nan=True, err_msg=f"{name}: {feature}", **tolerance)


@pytest.mark.filterwarnings("ignore")
def test_tsfeatures_instances_do_not_share_settings():
    x = np.array(REFERENCE["series"]["noise_200"])
    default = TsFeatures()
    TsFeatures(selected_features=["mean"])
    assert len(default.transform(x)) == 40


def test_default_features(unequal_series):
    wide = to_wide(calculate_features(unequal_series, "kats"))
    assert wide.shape == (3, 40)
    assert wide.index.tolist() == list(unequal_series)
    np.testing.assert_allclose(wide["kats__mean"], [x.mean() for x in unequal_series.values()])
    np.testing.assert_array_equal(wide["kats__length"], [len(x) for x in unequal_series.values()])


def test_options(unequal_series):
    def feature_names(**kwargs):
        return calculate_features(unequal_series, Kats(**kwargs))["feature"].unique().tolist()

    assert len(feature_names(nowcasting=True)) == 47
    assert feature_names(selected_features=["holt_params", "mean"]) == ["mean", "holt_alpha", "holt_beta"]
    excluded = feature_names(exclude_features=["hw_params", "hurst"])
    assert len(excluded) == 36 and "hurst" not in excluded and "hw_alpha" not in excluded


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"selected_features": ["cusum_detector"]}, "not available in pytheft"),
        ({"exclude_features": ["not_a_feature"]}, "Unknown Kats feature"),
        ({"selected_features": "mean"}, "must be a list"),
        ({"selected_features": []}, "at least one"),
    ],
)
def test_invalid_options(kwargs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        calculate_features(np.ones((1, 50)), Kats(**kwargs))


@pytest.mark.filterwarnings("ignore::pytheft.PyTheftWarning")
@pytest.mark.parametrize(("kwargs", "min_length"), [({}, 14), ({"stl_period": 12}, 24), ({"acfpacf_lag": 10}, 22)])
def test_series_of_the_minimum_length_can_be_processed(kwargs, min_length):
    rng = np.random.default_rng(0)
    data = {n: rng.standard_normal(n) + 3 for n in range(5, min_length + 15)}
    features = calculate_features(data, Kats(**kwargs))
    assert sorted(features["id"].unique()) == list(range(min_length, min_length + 15))


def test_short_series_are_skipped(unequal_series):
    with pytest.warns(PyTheftWarning, match=r"'kats' .* needs at least 14 values\): 'short'"):
        features = calculate_features({**unequal_series, "short": np.arange(12.0)}, "kats")
    assert set(features["id"]) == set(unequal_series)


def test_parallel_matches_sequential(unequal_series):
    sequential = calculate_features(unequal_series, "kats")
    parallel = calculate_features(unequal_series, "kats", n_jobs=2)
    pd.testing.assert_frame_equal(sequential, parallel)
