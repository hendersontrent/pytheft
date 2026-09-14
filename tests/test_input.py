import numpy as np
import pandas as pd
import pytest

from pytheft import PyTheftWarning
from pytheft._input import prepare_collection, to_collection


def test_2d_array_rows_are_series():
    X = np.arange(12, dtype=float).reshape(3, 4)
    collection = to_collection(X)
    assert len(collection) == 3
    np.testing.assert_array_equal(collection.series[1], X[1])
    assert collection.ids.tolist() == [0, 1, 2]
    assert collection.groups is None


def test_series_are_copies_of_the_input():
    X = np.zeros((2, 5))
    to_collection(X).series[0][0] = 1.0
    assert X[0, 0] == 0.0


def test_1d_input_is_a_single_series():
    assert len(to_collection(np.arange(5.0))) == 1
    assert len(to_collection([1.0, 2.0, 3.0])) == 1


def test_list_of_unequal_length_series_with_ids_and_groups():
    collection = to_collection([np.ones(5), [1, 2, 3]], ids=["a", "b"], groups=["g1", "g2"])
    assert [len(x) for x in collection.series] == [5, 3]
    assert collection.ids.tolist() == ["a", "b"]
    assert collection.groups.tolist() == ["g1", "g2"]


def test_dict_keys_are_ids():
    collection = to_collection({"x": np.ones(4), "y": np.zeros(7)})
    assert collection.ids.tolist() == ["x", "y"]
    assert [len(x) for x in collection.series] == [4, 7]


def test_pandas_series_of_arrays_uses_its_index_as_ids():
    data = pd.Series([np.ones(3), np.ones(4)], index=["a", "b"])
    assert to_collection(data).ids.tolist() == ["a", "b"]


def test_3d_array_with_one_channel():
    X = np.random.default_rng(0).standard_normal((4, 1, 20))
    np.testing.assert_array_equal(to_collection(X).series[2], X[2, 0])


def test_multivariate_3d_array_is_rejected():
    with pytest.raises(ValueError, match="univariate"):
        to_collection(np.zeros((4, 2, 20)))


def test_wide_dataframe_layout_reads_rows_as_series():
    df = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=["s1", "s2"])
    collection = to_collection(df, dataframe_layout="wide")
    assert collection.ids.tolist() == ["s1", "s2"]
    np.testing.assert_array_equal(collection.series[1], [3.0, 4.0, 5.0])


def test_long_dataframe_is_sorted_by_time_with_ids_in_order_of_appearance():
    df = pd.DataFrame(
        {
            "id": ["b", "b", "b", "a", "a"],
            "t": [2, 0, 1, 1, 0],
            "value": [2.0, 0.0, 1.0, 11.0, 10.0],
            "label": ["x", "x", "x", "y", "y"],
        }
    )
    collection = to_collection(df, time_col="t", group_col="label")
    assert collection.ids.tolist() == ["b", "a"]
    np.testing.assert_array_equal(collection.series[0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(collection.series[1], [10.0, 11.0])
    assert collection.groups.tolist() == ["x", "y"]


def test_long_dataframe_without_time_col_keeps_row_order():
    df = pd.DataFrame({"id": [1, 2, 1, 2], "value": [3.0, 9.0, 1.0, 8.0]})
    collection = to_collection(df)
    np.testing.assert_array_equal(collection.series[0], [3.0, 1.0])
    np.testing.assert_array_equal(collection.series[1], [9.0, 8.0])


def test_long_dataframe_with_datetime_times_and_custom_columns():
    times = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"sensor": ["s"] * 3, "when": times[::-1], "reading": [3.0, 2.0, 1.0]})
    collection = to_collection(df, id_col="sensor", time_col="when", value_col="reading")
    np.testing.assert_array_equal(collection.series[0], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("data", "kwargs", "error", "match"),
    [
        (pd.DataFrame({"x": [1.0]}), {}, ValueError, "to_numpy"),
        (pd.DataFrame({"id": [1, 1], "t": [0, 0], "value": [1.0, 2.0]}), {"time_col": "t"}, ValueError, "repeated time"),
        (pd.DataFrame({"id": [1, 1], "value": [1.0, 2.0], "g": ["a", "b"]}), {"group_col": "g"}, ValueError, "single value"),
        (pd.DataFrame({"id": [1], "value": [1.0]}), {"ids": ["a"]}, ValueError, "id_col"),
        (pd.DataFrame({"id": [None], "value": [1.0]}), {}, ValueError, "missing ids"),
        (np.ones((2, 5)), {"time_col": "t"}, ValueError, "long-format"),
        (np.ones((2, 5)), {"ids": ["a"]}, ValueError, "1 ids for 2"),
        (np.ones((2, 5)), {"ids": ["a", "a"]}, ValueError, "unique"),
        (np.ones((2, 5)), {"groups": ["a"]}, ValueError, "1 groups for 2"),
        ({"a": np.ones(3)}, {"ids": ["x"]}, ValueError, "keys are used as ids"),
        ([["a", "b"], ["c", "d"]], {}, TypeError, "converted to numbers"),
        ([np.ones((2, 2))], {}, ValueError, "one-dimensional"),
        ([], {}, ValueError, "No time series"),
        (np.ones((2, 2, 2, 2)), {}, ValueError, "must be 1D"),
    ],
)
def test_invalid_input(data, kwargs, error, match):
    with pytest.raises(error, match=match):
        to_collection(data, **kwargs)


def test_prepare_removes_empty_and_non_finite_series():
    collection = to_collection([np.ones(5), [1.0, np.nan, 2.0], [np.inf, 1.0], []], ids=list("abcd"))
    with pytest.warns(PyTheftWarning, match=r"Removed 3 of 4 time series .*'b', 'c', 'd'"):
        prepared, positions = prepare_collection(collection, z_score=False, on_invalid="remove")
    assert prepared.ids.tolist() == ["a"]
    assert positions.tolist() == [0]


def test_prepare_z_scores_with_ddof_1_and_sets_aside_constant_series():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    collection = to_collection([x, np.ones(4)])
    with pytest.warns(PyTheftWarning, match="Features are NaN for 1 of 2 .* cannot be z-scored"):
        prepared, positions = prepare_collection(collection, z_score=True, on_invalid="nan")
    np.testing.assert_allclose(prepared.series[0], (x - x.mean()) / x.std(ddof=1))
    assert positions.tolist() == [0]


def test_prepare_raises_when_no_series_remain():
    with pytest.warns(PyTheftWarning), pytest.raises(ValueError, match="No time series remain"):
        prepare_collection(to_collection([[np.nan, 1.0]]), z_score=False, on_invalid="remove")
