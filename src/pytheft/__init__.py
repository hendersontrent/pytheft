"""pytheft: Tools for Handling Extraction of Features from Time series, in Python.

pytheft provides one interface to several time-series feature sets (catch22,
tsfresh, TSFEL, Kats and hctsa), with one input format and one tidy output format.
"""

from pytheft._calculate import calculate_features, to_wide
from pytheft._transformer import FeatureExtractor
from pytheft._utils import PyTheftWarning
from pytheft.feature_sets import HCTSA, TSFEL, Catch22, FeatureSet, Kats, TSFresh

__version__ = "0.1.0"

__all__ = [
    "HCTSA",
    "TSFEL",
    "Catch22",
    "FeatureExtractor",
    "FeatureSet",
    "Kats",
    "PyTheftWarning",
    "TSFresh",
    "calculate_features",
    "to_wide",
]
