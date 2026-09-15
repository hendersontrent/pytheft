"""Checks that each feature set matches calling its library directly.

Tests for a library are skipped when it is not installed.
"""

import importlib.util
import subprocess
import sys
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from pytheft import HCTSA, TSFEL, Catch22, PyTheftWarning, TSFeatures, TSFresh, calculate_features, to_wide
from pytheft.feature_sets import _tsfeatures as _tsfeatures_module
from pytheft.feature_sets._hctsa import _clean_output


def test_catch22_matches_pycatch22(unequal_series):
    pycatch22 = pytest.importorskip("pycatch22")
    wide = to_wide(calculate_features(unequal_series, "catch22"))
    for key, x in unequal_series.items():
        expected = pycatch22.catch22_all(x.tolist())
        np.testing.assert_allclose(wide.loc[key].to_numpy(), expected["values"])
    assert wide.columns.tolist() == [f"catch22__{name}" for name in expected["names"]]


def test_catch24_adds_mean_and_sd(unequal_series):
    pytest.importorskip("pycatch22")
    wide = to_wide(calculate_features(unequal_series, Catch22(catch24=True)))
    assert wide.shape == (3, 24)
    np.testing.assert_allclose(wide["catch22__DN_Mean"], [x.mean() for x in unequal_series.values()])


def test_tsfresh_matches_tsfresh_in_input_order(unequal_series):
    pytest.importorskip("tsfresh")
    from tsfresh import extract_features

    wide = to_wide(calculate_features(unequal_series, "tsfresh"))
    long = pd.concat(
        pd.DataFrame({"id": key, "time": np.arange(len(x)), "value": x}) for key, x in unequal_series.items()
    )
    expected = extract_features(long, column_id="id", column_sort="time", n_jobs=0, disable_progressbar=True)
    expected.columns = ["tsfresh__" + column.removeprefix("value__") for column in expected.columns]

    assert wide.index.tolist() == ["b", "a", "c"]
    pd.testing.assert_frame_equal(wide, expected.loc[wide.index.tolist()], check_names=False)


def test_tsfresh_settings(unequal_series):
    pytest.importorskip("tsfresh")
    from tsfresh.feature_extraction import MinimalFCParameters

    features = calculate_features(unequal_series, TSFresh(settings="minimal"))
    assert features.groupby("id").size().eq(len(MinimalFCParameters())).all()

    with pytest.raises(ValueError, match="Unknown tsfresh settings"):
        calculate_features(unequal_series, TSFresh(settings="everything"))


@pytest.mark.filterwarnings("ignore:Using default sampling frequency")
def test_tsfel_matches_tsfel(unequal_series):
    tsfel = pytest.importorskip("tsfel")
    wide = to_wide(calculate_features(unequal_series, "tsfel"))
    expected = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain(), list(unequal_series.values()), verbose=0, n_jobs=None
    )
    expected.columns = ["tsfel__" + column.removeprefix("0_") for column in expected.columns]
    expected.index = wide.index
    pd.testing.assert_frame_equal(wide, expected, check_names=False)


def test_tsfel_domain_and_config_options(unequal_series):
    tsfel = pytest.importorskip("tsfel")
    statistical = tsfel.get_features_by_domain("statistical")["statistical"]
    features = calculate_features(unequal_series, TSFEL(domain="statistical"))
    assert all(any(name.startswith(s) for s in statistical) for name in features["feature"].unique())

    with pytest.raises(ValueError, match="either domain or config"):
        calculate_features(unequal_series, TSFEL(domain="statistical", config={}))


def test_tsfel_skips_series_that_are_too_short(unequal_series):
    pytest.importorskip("tsfel")
    with pytest.warns(PyTheftWarning, match="needs at least 12 values"):
        features = calculate_features({**unequal_series, "short": np.arange(8.0)}, "tsfel")
    assert set(features["id"]) == set(unequal_series)


def _import_tsfeatures():
    # pytest.importorskip would let tsfeatures disable warnings.warn for the rest of the session.
    if importlib.util.find_spec("tsfeatures") is None:
        pytest.skip("tsfeatures is not installed")
    return _tsfeatures_module._import_tsfeatures()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("freq", [1, 12])
def test_tsfeatures_matches_tsfeatures(unequal_series, freq):
    tsfeatures = _import_tsfeatures()
    wide = to_wide(calculate_features(unequal_series, TSFeatures(freq=freq)))
    long = pd.concat(
        pd.DataFrame({"unique_id": key, "ds": np.arange(len(x)), "y": x}) for key, x in unequal_series.items()
    )
    expected = tsfeatures.tsfeatures(long, freq=freq, threads=1).set_index("unique_id").loc[list(unequal_series)]
    expected.columns = ["tsfeatures__" + column for column in expected.columns]

    assert wide.shape == (3, 37 if freq == 1 else 42)
    assert wide.index.tolist() == list(unequal_series)
    pd.testing.assert_frame_equal(wide, expected[wide.columns].astype(float), check_names=False)


def test_tsfeatures_options(unequal_series):
    _import_tsfeatures()
    wide = to_wide(calculate_features(unequal_series, TSFeatures(features=["intervals", "statistics"])))
    assert wide.columns.tolist()[:4] == [f"tsfeatures__{name}" for name in ("intervals_mean", "intervals_sd", "total_sum", "mean")]
    # intervals modifies its input, which must not change the features computed after it.
    np.testing.assert_allclose(wide["tsfeatures__mean"], 0, atol=1e-12)

    unscaled = to_wide(calculate_features(unequal_series, TSFeatures(features=["statistics"], scale=False)))
    np.testing.assert_allclose(unscaled["tsfeatures__mean"], [x.mean() for x in unequal_series.values()])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"features": ["not_a_function"]}, ValueError, "Unknown tsfeatures function"),
        ({"features": "entropy"}, TypeError, "must be a list"),
        ({"features": []}, ValueError, "at least one"),
        ({"freq": 0}, ValueError, "positive integer"),
        ({"scale": "yes"}, TypeError, "True or False"),
    ],
)
def test_tsfeatures_invalid_options(kwargs, error, match):
    with pytest.raises(error, match=match):
        calculate_features(np.ones((1, 50)), TSFeatures(**kwargs))


def test_tsfeatures_skips_series_too_short_for_guerrero(unequal_series):
    _import_tsfeatures()
    with pytest.warns(PyTheftWarning, match=r"needs at least 13 values\): 'short'"):
        features = calculate_features({**unequal_series, "short": np.arange(1.0, 13.0)}, TSFeatures(["guerrero"], 12))
    assert set(features["id"]) == set(unequal_series)


def test_tsfeatures_import_leaves_the_process_unchanged():
    if importlib.util.find_spec("tsfeatures") is None:
        pytest.skip("tsfeatures is not installed")
    # A new process, because tsfeatures changes the process only the first time it is imported.
    code = (
        "import os, warnings, numpy as np\n"
        "from pytheft.feature_sets._tsfeatures import _import_tsfeatures\n"
        "before = (warnings.warn, np.geterr(), os.environ.get('OMP_NUM_THREADS'))\n"
        "_import_tsfeatures()\n"
        "assert before == (warnings.warn, np.geterr(), os.environ.get('OMP_NUM_THREADS'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tsfeatures_parallel_matches_sequential(unequal_series):
    _import_tsfeatures()
    sequential = calculate_features(unequal_series, "tsfeatures")
    parallel = calculate_features(unequal_series, "tsfeatures", n_jobs=2)
    pd.testing.assert_frame_equal(sequential, parallel)


def _hctsa_module_config(module: str) -> str:
    return str(files("pyhctsa") / "configurations" / "module_configs" / f"{module}.yaml")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_hctsa_matches_pyhctsa(unequal_series):
    pytest.importorskip("pyhctsa")
    from pyhctsa.calculator import FeatureCalculator

    config = _hctsa_module_config("distribution")
    wide = to_wide(calculate_features(unequal_series, HCTSA(config_path=config)))
    expected = FeatureCalculator(config_path=config).extract(list(unequal_series.values()))
    numeric = expected.select_dtypes("number")

    assert wide.index.tolist() == list(unequal_series)
    assert numeric.shape[1] > 0
    for column in numeric.columns:
        np.testing.assert_allclose(wide[f"hctsa__{column}"], numeric[column].astype(float), equal_nan=True)


def test_hctsa_skips_short_and_constant_series(unequal_series):
    pytest.importorskip("pyhctsa")
    data = {**unequal_series, "short": np.random.default_rng(0).standard_normal(50), "flat": np.ones(150)}
    with pytest.warns(PyTheftWarning, match=r"needs at least 100 values and not constant\): 'short', 'flat'"):
        features = calculate_features(data, HCTSA(config_path=_hctsa_module_config("distribution")))
    assert set(features["id"]) == set(unequal_series)


def test_hctsa_default_configuration():
    pytest.importorskip("pyhctsa")
    x = np.random.default_rng(1).standard_normal(120)
    features = calculate_features([x], "hctsa")
    assert features["feature"].nunique() > 5000
    assert features["value"].dtype == float


def test_hctsa_missing_config_file():
    with pytest.raises(FileNotFoundError):
        calculate_features(np.ones((1, 200)), HCTSA(config_path="no/such/config.yaml"))


def test_hctsa_output_cleaning():
    raw = pd.DataFrame(
        {
            "op_a": [1, 2],
            "op_b": [np.array([0.5]), "Error: failed"],
            "op_c": ["Error: failed", np.nan],
            "op_c.x": [np.nan, 3.0],
            "op_d": [complex(1, 0), complex(1, 1)],
            "op_e": ["Error: failed", "Error: failed"],
        }
    )
    expected = pd.DataFrame(
        {
            "op_a": [1.0, 2.0],
            "op_b": [0.5, np.nan],
            "op_c.x": [np.nan, 3.0],
            "op_d": [1.0, np.nan],
            "op_e": [np.nan, np.nan],
        }
    )
    pd.testing.assert_frame_equal(_clean_output(raw), expected)
