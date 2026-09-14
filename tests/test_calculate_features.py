import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from pytheft import FeatureSet, PyTheftWarning, calculate_features, to_wide


class Stats(FeatureSet):
    name = "stats"

    def _calculate(self, series, *, n_jobs, verbose):
        return pd.DataFrame(
            {
                "mean": [x.mean() for x in series],
                "length": [len(x) for x in series],
                "first": [x[0] for x in series],
            }
        )


class NeedsTenValues(FeatureSet):
    name = "needs_ten"

    def _supported(self, series):
        return np.array([len(x) >= 10 for x in series])

    def _requirements(self):
        return "at least 10 values"

    def _calculate(self, series, *, n_jobs, verbose):
        return pd.DataFrame({"max": [x.max() for x in series]})


class Noisy(FeatureSet):
    name = "noisy"

    def _calculate(self, series, *, n_jobs, verbose):
        print("library chatter")
        warnings.warn("library warning", RuntimeWarning)
        logging.getLogger("some.library").warning("library log")
        return pd.DataFrame({"length": [len(x) for x in series]})


class NotInstalled(FeatureSet):
    name = "missing"
    _required_modules = ("pytheft_module_that_does_not_exist",)
    _extra = "missing"

    def _calculate(self, series, *, n_jobs, verbose):
        raise AssertionError("should not be called")


def test_output_is_tidy(simulated):
    X, y = simulated
    features = calculate_features(X, Stats(), groups=y)
    assert features.columns.tolist() == ["id", "group", "feature_set", "feature", "value"]
    assert len(features) == 3 * len(X)
    assert (features["feature_set"] == "stats").all()

    second = features[features["id"] == 1]
    assert second["feature"].tolist() == ["mean", "length", "first"]
    assert second["group"].unique().tolist() == [y[1]]
    assert second["value"].tolist() == pytest.approx([X[1].mean(), X.shape[1], X[1, 0]])


def test_group_column_only_when_groups_are_given(simulated):
    assert "group" not in calculate_features(simulated[0], Stats()).columns


def test_user_features(simulated):
    X, _ = simulated
    features = calculate_features(X, feature_set=None, features={"mean": np.mean, "sd": np.std})
    assert set(features["feature_set"]) == {"user"}
    np.testing.assert_allclose(to_wide(features)["user__sd"], X.std(axis=1))


def test_user_feature_must_return_a_number():
    with pytest.raises(TypeError, match="single number"):
        calculate_features(np.ones((2, 10)), None, features={"bad": lambda x: x[:2]})


def test_user_features_must_be_a_dict_of_functions():
    with pytest.raises(TypeError, match="non-empty dict"):
        calculate_features(np.ones((2, 10)), None, features=[np.mean])
    with pytest.raises(TypeError, match="must be a function"):
        calculate_features(np.ones((2, 10)), None, features={"mean": "np.mean"})


def test_feature_sets_are_returned_in_the_order_requested(simulated):
    features = calculate_features(simulated[0], [NeedsTenValues(), Stats()], features={"median": np.median})
    assert features["feature_set"].unique().tolist() == ["needs_ten", "stats", "user"]


def test_long_dataframe_matches_array_input(simulated):
    X, y = simulated
    ids = [f"s{i}" for i in range(len(X))]
    long = pd.DataFrame(
        {
            "series": np.repeat(ids, X.shape[1]),
            "timepoint": np.tile(np.arange(X.shape[1]), len(X)),
            "values": X.ravel(),
            "process": np.repeat(y, X.shape[1]),
        }
    ).sample(frac=1, random_state=0)

    from_long = calculate_features(
        long, Stats(), id_col="series", time_col="timepoint", value_col="values", group_col="process"
    )
    from_array = calculate_features(X, Stats(), ids=ids, groups=y)
    pd.testing.assert_frame_equal(to_wide(from_long).sort_index(), to_wide(from_array).sort_index())


def test_series_with_missing_values_are_removed(simulated):
    X = simulated[0].copy()
    X[2, 5] = np.nan
    with pytest.warns(PyTheftWarning, match="Removed 1 of 12 time series"):
        features = calculate_features(X, Stats())
    assert 2 not in set(features["id"])


def test_unsupported_series_are_skipped_only_for_that_feature_set():
    data = {"short": np.arange(5.0), "long": np.arange(20.0)}
    with pytest.warns(PyTheftWarning, match=r"'needs_ten' .* needs at least 10 values\): 'short'"):
        features = calculate_features(data, [Stats(), NeedsTenValues()])
    counts = features.groupby(["feature_set", "id"]).size()
    assert counts[("stats", "short")] == 3
    assert ("needs_ten", "short") not in counts.index


def test_z_score(simulated):
    features = calculate_features(
        simulated[0], None, features={"mean": np.mean, "sd": lambda x: np.std(x, ddof=1)}, z_score=True
    )
    wide = to_wide(features)
    np.testing.assert_allclose(wide["user__mean"], 0, atol=1e-12)
    np.testing.assert_allclose(wide["user__sd"], 1)


def test_library_output_is_silenced_unless_verbose(capsys, caplog):
    X = np.arange(10.0).reshape(2, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_features(X, Noisy())
    assert not caught
    assert not caplog.records
    assert "library chatter" not in capsys.readouterr().out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_features(X, Noisy(), verbose=True)
    assert [str(w.message) for w in caught] == ["library warning"]
    assert [r.getMessage() for r in caplog.records] == ["library log"]
    assert "library chatter" in capsys.readouterr().out


def test_feature_set_names_are_case_insensitive():
    pytest.importorskip("pycatch22")
    features = calculate_features(np.random.default_rng(0).standard_normal((2, 50)), "CATCH22")
    assert set(features["feature_set"]) == {"catch22"}


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"feature_set": "feasts"}, ValueError, "choose from catch22, hctsa, kats, tsfel, tsfresh"),
        ({"feature_set": None}, ValueError, "No features requested"),
        ({"feature_set": [Stats(), Stats()]}, ValueError, "only be requested once"),
        ({"feature_set": [np.mean]}, TypeError, "feature_set entries"),
        ({"feature_set": [Stats(), NotInstalled()]}, ImportError, r'pip install "pytheft\[missing\]"'),
        ({"feature_set": Stats(), "n_jobs": 0}, ValueError, "n_jobs"),
    ],
)
def test_invalid_arguments(kwargs, error, match):
    with pytest.raises(error, match=match):
        calculate_features(np.ones((2, 10)), **kwargs)


def test_to_wide_keeps_input_order_and_groups():
    features = calculate_features({"b": np.arange(10.0), "a": np.arange(20.0)}, Stats(), groups=["g1", "g2"])
    wide = to_wide(features)
    assert wide.index.names == ["id", "group"]
    assert wide.index.tolist() == [("b", "g1"), ("a", "g2")]
    assert wide.columns.tolist() == ["stats__mean", "stats__length", "stats__first"]


def test_to_wide_requires_tidy_columns():
    with pytest.raises(ValueError, match="missing the columns"):
        to_wide(pd.DataFrame({"id": [1]}))
