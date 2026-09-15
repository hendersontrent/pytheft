"""Feature sets that pytheft can compute."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from pytheft.feature_sets._base import FeatureSet
from pytheft.feature_sets._catch22 import Catch22
from pytheft.feature_sets._hctsa import HCTSA
from pytheft.feature_sets._kats import Kats
from pytheft.feature_sets._tsfeatures import TSFeatures
from pytheft.feature_sets._tsfel import TSFEL
from pytheft.feature_sets._tsfresh import TSFresh
from pytheft.feature_sets._user import UserFeatures

__all__ = ["HCTSA", "TSFEL", "Catch22", "FeatureSet", "Kats", "TSFeatures", "TSFresh"]

_BY_NAME: dict[str, type[FeatureSet]] = {cls.name: cls for cls in (Catch22, HCTSA, Kats, TSFEL, TSFeatures, TSFresh)}


def resolve_feature_sets(
    feature_set: str | FeatureSet | Sequence[str | FeatureSet] | None,
    features: Mapping[str, Callable[[np.ndarray], float]] | None = None,
) -> list[FeatureSet]:
    """Turn the ``feature_set`` and ``features`` arguments into validated, installed feature sets."""
    if feature_set is None:
        requested = []
    elif isinstance(feature_set, (str, FeatureSet)):
        requested = [feature_set]
    else:
        requested = list(feature_set)

    feature_sets = [_resolve_one(item) for item in requested]
    if features is not None:
        feature_sets.append(UserFeatures(features))
    if not feature_sets:
        raise ValueError("No features requested: pass feature_set, features, or both.")

    names = [fs.name for fs in feature_sets]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Each feature set can only be requested once; repeated: {', '.join(repeated)}.")

    for fs in feature_sets:
        fs._validate_params()
    for fs in feature_sets:
        fs._check_installed()
    return feature_sets


def _resolve_one(item: str | FeatureSet) -> FeatureSet:
    if isinstance(item, FeatureSet):
        return item
    if isinstance(item, str):
        cls = _BY_NAME.get(item.lower())
        if cls is None:
            raise ValueError(f"Unknown feature set {item!r}; choose from {', '.join(sorted(_BY_NAME))}.")
        return cls()
    raise TypeError(
        f"feature_set entries must be names such as 'catch22' or FeatureSet instances, got {type(item).__name__}."
    )
