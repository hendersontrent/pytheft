"""Conversion of the supported time-series inputs into one internal representation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from pytheft._utils import format_ids, warn


@dataclass(frozen=True)
class TimeSeriesCollection:
    """Univariate time series with their ids and optional group labels.

    Every input format accepted by pytheft is converted to this, so feature
    sets only ever receive a list of one-dimensional float arrays.
    """

    series: list[np.ndarray]
    ids: pd.Index
    groups: pd.Index | None = None

    def __len__(self) -> int:
        return len(self.series)

    def take(self, positions: np.ndarray) -> TimeSeriesCollection:
        """Return the time series at the given positions."""
        return TimeSeriesCollection(
            series=[self.series[i] for i in positions],
            ids=self.ids.take(positions),
            groups=None if self.groups is None else self.groups.take(positions),
        )


def to_collection(
    data,
    *,
    ids=None,
    groups=None,
    id_col: str = "id",
    time_col: str | None = None,
    value_col: str = "value",
    group_col: str | None = None,
    dataframe_layout: Literal["long", "wide"] = "long",
) -> TimeSeriesCollection:
    """Convert supported input data into a :class:`TimeSeriesCollection`.

    ``dataframe_layout`` controls how a DataFrame is read: ``"long"`` (one row
    per observation, as in :func:`pytheft.calculate_features`) or ``"wide"``
    (one row per time series, the scikit-learn convention).

    The series are copied, so feature libraries cannot modify the caller's data.
    """
    if isinstance(data, pd.DataFrame) and dataframe_layout == "long":
        if ids is not None or groups is not None:
            raise ValueError(
                "For DataFrame input, identify time series and groups with id_col and "
                "group_col rather than ids and groups."
            )
        return _from_long_dataframe(
            data, id_col=id_col, time_col=time_col, value_col=value_col, group_col=group_col
        )

    if time_col is not None or group_col is not None:
        raise ValueError("time_col and group_col only apply to long-format DataFrame input.")

    if isinstance(data, pd.DataFrame):
        series = list(_as_matrix(data))
        # Rows of a wide DataFrame are matched by position, so a repeated index is fine.
        default_ids = data.index if data.index.is_unique else pd.RangeIndex(len(data))
    elif isinstance(data, Mapping):
        if ids is not None:
            raise ValueError("For dict input, the keys are used as ids, so ids cannot also be given.")
        series = [_as_series(values, key) for key, values in data.items()]
        default_ids = pd.Index(list(data.keys()))
    else:
        series, default_ids = _from_array_like(data)

    if not series:
        raise ValueError("No time series were supplied.")

    ids = default_ids if ids is None else pd.Index(ids)
    if len(ids) != len(series):
        raise ValueError(f"Got {len(ids)} ids for {len(series)} time series.")
    if not ids.is_unique:
        raise ValueError(f"ids must be unique; repeated: {format_ids(ids[ids.duplicated()].unique())}.")

    if groups is not None:
        groups = pd.Index(groups)
        if len(groups) != len(series):
            raise ValueError(f"Got {len(groups)} groups for {len(series)} time series.")

    return TimeSeriesCollection(series=series, ids=ids, groups=groups)


def prepare_collection(
    collection: TimeSeriesCollection,
    *,
    z_score: bool,
    on_invalid: Literal["remove", "nan"],
) -> tuple[TimeSeriesCollection, np.ndarray]:
    """Set aside time series that features cannot be computed on, then optionally z-score.

    Returns the usable time series and their positions in ``collection``.
    ``on_invalid`` only changes the wording of the warning: callers either
    drop the unusable series or report NaN features for them.
    """
    usable = np.array([x.size > 0 and bool(np.isfinite(x).all()) for x in collection.series])
    _warn_unusable(collection, ~usable, "are empty or contain missing or non-finite values", on_invalid)

    if z_score:
        scalable = np.array([x.size > 1 and bool(np.ptp(x) > 0) for x in collection.series])
        _warn_unusable(
            collection,
            usable & ~scalable,
            "are constant or have fewer than two values, so cannot be z-scored",
            on_invalid,
        )
        usable &= scalable

    positions = np.flatnonzero(usable)
    if positions.size == 0:
        raise ValueError("No time series remain that features can be calculated on.")

    prepared = collection.take(positions)
    if z_score:
        prepared = replace(prepared, series=[(x - x.mean()) / x.std(ddof=1) for x in prepared.series])
    return prepared, positions


def _warn_unusable(
    collection: TimeSeriesCollection,
    mask: np.ndarray,
    reason: str,
    on_invalid: Literal["remove", "nan"],
) -> None:
    if not mask.any():
        return
    count = f"{int(mask.sum())} of {len(collection)} time series"
    action = f"Removed {count}" if on_invalid == "remove" else f"Features are NaN for {count}"
    warn(f"{action} that {reason}: {format_ids(collection.ids[mask])}.")


def _from_array_like(data) -> tuple[list[np.ndarray], pd.Index]:
    if isinstance(data, pd.Series):
        if data.dtype == object and len(data) and np.ndim(data.iloc[0]) > 0:
            return [_as_series(values, key) for key, values in data.items()], data.index
        return [_as_series(data, 0)], pd.RangeIndex(1)

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [], pd.RangeIndex(0)
        if all(np.ndim(item) == 0 for item in data):
            return [_as_series(data, 0)], pd.RangeIndex(1)
        return [_as_series(item, i) for i, item in enumerate(data)], pd.RangeIndex(len(data))

    array = np.asarray(data)
    if array.dtype == object and array.ndim == 1 and array.size and np.ndim(array[0]) > 0:
        return [_as_series(item, i) for i, item in enumerate(array)], pd.RangeIndex(len(array))
    if array.ndim == 1:
        return [_as_series(array, 0)], pd.RangeIndex(1)
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    elif array.ndim == 3:
        raise ValueError(
            f"Got a 3D array of shape {array.shape}. pytheft computes features on univariate "
            "time series, so 3D input must have shape (n_series, 1, n_timepoints)."
        )
    if array.ndim != 2:
        raise ValueError(
            "Array input must be 1D (one time series) or 2D (n_series, n_timepoints), "
            f"got shape {array.shape}."
        )
    return list(_as_matrix(array)), pd.RangeIndex(len(array))


def _from_long_dataframe(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str | None,
    value_col: str,
    group_col: str | None,
) -> TimeSeriesCollection:
    arguments = {"id_col": id_col, "value_col": value_col, "time_col": time_col, "group_col": group_col}
    missing = {arg: col for arg, col in arguments.items() if col is not None and col not in df.columns}
    if missing:
        details = ", ".join(f"{arg}={col!r}" for arg, col in missing.items())
        raise ValueError(
            "DataFrame input is read as long format (one row per observation), but these "
            f"columns were not found: {details}. Available columns: {list(df.columns)}. "
            "If each row of your DataFrame is a separate time series, pass df.to_numpy() instead."
        )
    if df.empty:
        raise ValueError("The DataFrame has no rows.")
    if df[id_col].isna().any():
        raise ValueError(f"Column {id_col!r} contains missing ids.")

    codes, uniques = pd.factorize(df[id_col], sort=False)
    values = _as_series(df[value_col], value_col)

    if time_col is None:
        order = np.argsort(codes, kind="stable")
    else:
        if df[time_col].isna().any():
            raise ValueError(f"Column {time_col!r} contains missing time values.")
        keys = pd.DataFrame({"code": codes, "time": df[time_col].to_numpy()})
        order = keys.sort_values(["code", "time"]).index.to_numpy()
        sorted_codes = codes[order]
        sorted_times = keys["time"].to_numpy()[order]
        repeated = (sorted_codes[1:] == sorted_codes[:-1]) & np.asarray(
            sorted_times[1:] == sorted_times[:-1], dtype=bool
        )
        if repeated.any():
            example = uniques[sorted_codes[1:][repeated][0]]
            raise ValueError(
                f"Column {time_col!r} has repeated time values within a time series "
                f"(for example, id {format_ids([example])})."
            )

    counts = np.bincount(codes, minlength=len(uniques))
    starts = np.cumsum(counts) - counts
    series = np.split(values[order], starts[1:])

    groups = None
    if group_col is not None:
        n_groups = df[group_col].groupby(codes).nunique(dropna=False)
        if (n_groups > 1).any():
            example = uniques[n_groups.index[n_groups > 1][0]]
            raise ValueError(
                f"Column {group_col!r} must have a single value within each time series, "
                f"but id {format_ids([example])} has several."
            )
        groups = pd.Index(df[group_col].iloc[order[starts]].array)

    return TimeSeriesCollection(series=series, ids=pd.Index(uniques), groups=groups)


def _as_matrix(data) -> np.ndarray:
    try:
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(dtype=float, na_value=np.nan, copy=True)
        return np.array(data, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Time series values could not be converted to numbers: {error}") from None


def _as_series(values, label) -> np.ndarray:
    try:
        if isinstance(values, (pd.Series, pd.Index)):
            array = values.to_numpy(dtype=float, na_value=np.nan, copy=True)
        else:
            array = np.array(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Time series {format_ids([label])} could not be converted to numbers: {error}"
        ) from None
    if array.ndim != 1:
        raise ValueError(
            f"Time series {format_ids([label])} must be one-dimensional, got shape {array.shape}."
        )
    return array
