"""The functional interface: :func:`calculate_features` and :func:`to_wide`."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from pytheft._input import TimeSeriesCollection, prepare_collection, to_collection
from pytheft._utils import effective_n_jobs, format_ids, quiet, warn
from pytheft.feature_sets import FeatureSet, resolve_feature_sets

_OUTPUT_COLUMNS = ["id", "group", "feature_set", "feature", "value"]


def calculate_features(
    data,
    feature_set: str | FeatureSet | Sequence[str | FeatureSet] | None = "catch22",
    *,
    features: Mapping[str, Callable[[np.ndarray], float]] | None = None,
    ids=None,
    groups=None,
    id_col: str = "id",
    time_col: str | None = None,
    value_col: str = "value",
    group_col: str | None = None,
    z_score: bool = False,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate time-series features and return them as a tidy DataFrame.

    Parameters
    ----------
    data : array-like, list, dict or DataFrame
        The time series, in any of these formats:

        - a 2D array of shape ``(n_series, n_timepoints)``, one series per row
          (the scikit-learn convention), or a 3D array of shape
          ``(n_series, 1, n_timepoints)``;
        - a 1D array, for a single series;
        - a list of 1D arrays, or a dict mapping ids to 1D arrays, for series
          of unequal length;
        - a long-format DataFrame with one row per observation (like a
          tsibble), described by ``id_col``, ``value_col`` and optionally
          ``time_col`` and ``group_col``.
    feature_set : str, FeatureSet or list of these, default="catch22"
        Feature sets to compute: any of ``"catch22"``, ``"tsfresh"``,
        ``"tsfel"`` and ``"hctsa"`` (case-insensitive), or configured
        instances such as ``Catch22(catch24=True)``. Pass ``None`` to compute
        only the functions in ``features``.
    features : dict, optional
        Your own features, as a dict mapping feature names to functions that
        take a 1D NumPy array and return a number, e.g. ``{"mean": np.mean}``.
        They are labelled with the feature set ``"user"``.
    ids : array-like, optional
        One id per series, for array and list input. Defaults to positions
        ``0, 1, ...`` (dict keys for dict input).
    groups : array-like, optional
        One group label (such as a class) per series, for array, list and dict
        input. Adds a ``group`` column to the output.
    id_col, time_col, value_col, group_col : str, optional
        Column names for long-format DataFrame input. Without ``time_col``,
        observations are used in row order within each series.
    z_score : bool, default=False
        Standardise each series to mean 0 and standard deviation 1 (with
        ``ddof=1``, as in R's ``scale``) before computing features.
    n_jobs : int, optional
        Number of processes for feature sets that support parallelism
        (tsfresh, TSFEL and hctsa). ``None`` means 1 and ``-1`` means all
        CPUs. Scripts using more than one process need an
        ``if __name__ == "__main__":`` guard.
    verbose : bool, default=False
        Print progress and show the feature libraries' own output and
        warnings, which are otherwise silenced.

    Returns
    -------
    pandas.DataFrame
        One row per feature per series, with columns ``id``, ``group`` (only
        when groups are given), ``feature_set``, ``feature`` and ``value``.
        Series containing missing or non-finite values are removed with a
        :class:`PyTheftWarning`, as are series a feature set cannot process
        (for example, hctsa needs at least 100 values).

    See Also
    --------
    to_wide : Reshape the result into one row per series.
    FeatureExtractor : The same computation as a scikit-learn transformer.

    Examples
    --------
    >>> import numpy as np
    >>> from pytheft import calculate_features
    >>> X = np.random.default_rng(0).standard_normal((10, 200))
    >>> features = calculate_features(X, feature_set=None, features={"mean": np.mean})
    >>> features.columns.tolist()
    ['id', 'feature_set', 'feature', 'value']
    """
    feature_sets = resolve_feature_sets(feature_set, features)
    n_jobs = effective_n_jobs(n_jobs)
    collection = to_collection(
        data,
        ids=ids,
        groups=groups,
        id_col=id_col,
        time_col=time_col,
        value_col=value_col,
        group_col=group_col,
    )
    collection, _ = prepare_collection(collection, z_score=z_score, on_invalid="remove")
    results = compute_feature_sets(collection, feature_sets, n_jobs=n_jobs, verbose=verbose)
    return _to_long(collection, results)


def to_wide(features: pd.DataFrame) -> pd.DataFrame:
    """Reshape the output of :func:`calculate_features` to one row per time series.

    Parameters
    ----------
    features : pandas.DataFrame
        Tidy features as returned by :func:`calculate_features`.

    Returns
    -------
    pandas.DataFrame
        One column per feature, named ``"{feature_set}__{feature}"``, in the
        order computed. The index is ``id``, or ``(id, group)`` when groups
        are present so class labels stay aligned with the rows.
    """
    missing = {"id", "feature_set", "feature", "value"} - set(features.columns)
    if missing:
        raise ValueError(f"features is missing the columns: {', '.join(sorted(missing))}.")

    index_columns = ["id", "group"] if "group" in features.columns else ["id"]
    column = features["feature_set"].astype(str) + "__" + features["feature"].astype(str)
    wide = features.assign(_column=column).pivot(
        index=index_columns if len(index_columns) > 1 else "id",
        columns="_column",
        values="value",
    )
    if len(index_columns) > 1:
        rows = pd.MultiIndex.from_frame(features[index_columns].drop_duplicates())
    else:
        rows = pd.Index(features["id"].unique(), name="id")
    wide = wide.reindex(index=rows, columns=column.unique())
    wide.columns.name = None
    return wide


def compute_feature_sets(
    collection: TimeSeriesCollection,
    feature_sets: list[FeatureSet],
    *,
    n_jobs: int,
    verbose: bool,
) -> list[tuple[FeatureSet, pd.DataFrame]]:
    """Compute each feature set on the series it supports.

    Each result has one row per computed series, indexed by the series'
    position in ``collection``, and one float column per feature.
    """
    results = []
    for feature_set in feature_sets:
        if verbose:
            print(f"Computing {feature_set.name} features...")
        with quiet(not verbose):
            # Checking requirements can import the library, which may itself be noisy.
            supported = np.asarray(feature_set._supported(collection.series), dtype=bool)
            requirements = "" if supported.all() else feature_set._requirements()
        if not supported.all():
            skipped = collection.ids[~supported]
            warn(
                f"Skipped {len(skipped)} of {len(collection)} time series that the {feature_set.name!r} "
                f"feature set cannot process{f' (it needs {requirements})' if requirements else ''}: "
                f"{format_ids(skipped)}."
            )
        positions = np.flatnonzero(supported)
        if positions.size == 0:
            results.append((feature_set, pd.DataFrame(index=pd.Index([], dtype=int))))
            continue

        with quiet(not verbose):
            computed = feature_set._calculate(
                [collection.series[i] for i in positions], n_jobs=n_jobs, verbose=verbose
            )
        results.append((feature_set, _check_result(feature_set, computed, positions)))
    return results


def _check_result(feature_set: FeatureSet, computed: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    if not isinstance(computed, pd.DataFrame):
        raise TypeError(f"The {feature_set.name!r} feature set returned {type(computed).__name__}, not a DataFrame.")
    if len(computed) != len(positions):
        raise RuntimeError(
            f"The {feature_set.name!r} feature set returned {len(computed)} rows for {len(positions)} time series."
        )
    if not computed.columns.is_unique:
        raise RuntimeError(f"The {feature_set.name!r} feature set returned repeated feature names.")
    try:
        values = computed.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"The {feature_set.name!r} feature set returned non-numeric values: {error}") from None
    return pd.DataFrame(values, index=positions, columns=[str(column) for column in computed.columns])


def _to_long(collection: TimeSeriesCollection, results: list[tuple[FeatureSet, pd.DataFrame]]) -> pd.DataFrame:
    columns = [c for c in _OUTPUT_COLUMNS if c != "group" or collection.groups is not None]
    frames = []
    for feature_set, computed in results:
        n_series, n_features = computed.shape
        if n_series == 0 or n_features == 0:
            continue
        positions = computed.index.to_numpy()
        frame = {"id": collection.ids.take(positions).repeat(n_features)}
        if collection.groups is not None:
            frame["group"] = collection.groups.take(positions).repeat(n_features)
        frame["feature_set"] = np.full(n_series * n_features, feature_set.name, dtype=object)
        frame["feature"] = np.tile(computed.columns.to_numpy(dtype=object), n_series)
        frame["value"] = computed.to_numpy().ravel()
        frames.append(pd.DataFrame(frame))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]
