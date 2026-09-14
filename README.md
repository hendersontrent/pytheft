# pytheft

Tools for Handling Extraction of Features from Time series (`theft`), in Python.

`pytheft` is the Python counterpart of the R package [theft](https://github.com/hendersontrent/theft). `pytheft` provides a singular, standardised interface to the following time-series feature sets:

- [catch22](https://github.com/DynamicsAndNeuralSystems/pycatch22): 22 features
- [tsfresh](https://tsfresh.com): 783 features
- [TSFEL](https://tsfel.readthedocs.io): 156 features
- [Kats](https://github.com/facebookresearch/Kats): 40 features
- [hctsa](https://github.com/DynamicsAndNeuralSystems/pyhctsa): over 5,000 features

Support is also provided for users to supply their own feature calculation functions.

## Installation

You can install the stable version of `pytheft` from GitHub using:

``` python
 pip install git+https://github.com/hendersontrent/pytheft
```

## Quick start

Each row of `X` is a time series:

```python
import numpy as np
from pytheft import calculate_features

rng = np.random.default_rng(0)
X = np.vstack([rng.standard_normal((20, 200)), np.cumsum(rng.standard_normal((20, 200)), axis=1)])
y = np.repeat(["noise", "random_walk"], 20)

features = calculate_features(X, feature_set="catch22", groups=y)
features.head()
```

```
   id  group feature_set                   feature     value
0   0  noise     catch22        DN_HistogramMode_5 -0.221238
1   0  noise     catch22       DN_HistogramMode_10  0.463797
2   0  noise     catch22                 CO_f1ecac  0.657821
3   0  noise     catch22            CO_FirstMin_ac  1.000000
4   0  noise     catch22  CO_HistogramAMI_even_2_5  0.066011
```

## Input formats

The main `calculate_features` functions accepts the following time-series data formats:

| Input format | Description |
| --- | --- |
| 2D array, `(n_series, n_timepoints)` | One series per row. Use `ids=` and `groups=` to specify rows. |
| 3D array, `(n_series, 1, n_timepoints)` | Univariate layout used by `aeon` and `sktime`. |
| List of 1D arrays, or dict of `id: array` | For series with different lengths. |
| Long-format DataFrame | One row per observation as per 'tidy' data principles and used by the `tsibble` package in R. |

## scikit-learn

`FeatureExtractor` computes the same features as a scikit-learn transformer, so a feature set can be part of a pipeline and cross-validated:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pytheft import FeatureExtractor

pipeline = make_pipeline(FeatureExtractor(feature_set="catch22"), StandardScaler(), LogisticRegression())
cross_val_score(pipeline, X, y, cv=5)
```

## Parallel processing and messages

The `n_jobs` argument sets the number of processes used by `tsfresh`, `tsfel`, `kats`, and `hctsa`.

## Citation

If you use `pytheft` in your own work, please cite the
paper:

T. Henderson and Ben D. Fulcher. [“Feature-Based Time-Series Analysis in
R using the Theft
Ecosystem”](https://journal.r-project.org/articles/RJ-2025-023/), The R
Journal, 2025.

BibTeX version:

    @article{RJ-2025-023,
      author = {Henderson, Trent and Fulcher, Ben D.},
      title = {Feature-Based Time-Series Analysis in R using the Theft Ecosystem},
      journal = {The R Journal},
      year = {2025},
      note = {https://doi.org/10.32614/RJ-2025-023},
      doi = {10.32614/RJ-2025-023},
      volume = {17},
      issue = {3},
      issn = {2073-4859},
      pages = {43-68}
    }
