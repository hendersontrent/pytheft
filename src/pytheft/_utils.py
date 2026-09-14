"""Internal helpers shared across pytheft."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Iterator, Sequence

import numpy as np

_PACKAGE_DIR = os.path.dirname(__file__)


class PyTheftWarning(UserWarning):
    """Warning about the input data, such as time series that were removed or skipped."""


def warn(message: str) -> None:
    """Emit a :class:`PyTheftWarning` attributed to the caller's code."""
    if sys.version_info >= (3, 12):
        warnings.warn(message, PyTheftWarning, skip_file_prefixes=(_PACKAGE_DIR,))
    else:
        warnings.warn(message, PyTheftWarning, stacklevel=3)


def format_ids(ids: Sequence, limit: int = 5) -> str:
    """Format a few ids for a message, e.g. ``'a', 'b', 'c', ...``."""
    ids = list(ids)
    shown = ", ".join(repr(i.item() if isinstance(i, np.generic) else i) for i in ids[:limit])
    return f"{shown}, ..." if len(ids) > limit else shown


def effective_n_jobs(n_jobs: int | None) -> int:
    """Resolve ``n_jobs`` following the joblib convention (``None`` is 1, ``-1`` is all CPUs)."""
    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs == 0:
        raise ValueError(f"n_jobs must be a non-zero integer or None, got {n_jobs!r}.")
    if n_jobs < 0:
        return max((os.cpu_count() or 1) + 1 + int(n_jobs), 1)
    return int(n_jobs)


@contextlib.contextmanager
def quiet(enabled: bool = True) -> Iterator[None]:
    """Silence warnings, printed output and log messages below ERROR while active.

    Feature libraries are very noisy (tsfresh and statsmodels emit many
    RuntimeWarnings, pyhctsa prints progress), so their calls are wrapped in
    this unless the user asks for verbose output. Worker processes started
    meanwhile inherit ``PYTHONWARNINGS=ignore``. Messages written directly to
    the C-level stderr stream, or logged by worker processes, still appear.
    """
    if not enabled:
        yield
        return
    previous_disable = logging.root.manager.disable
    previous_env = os.environ.get("PYTHONWARNINGS")
    with (
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        os.environ["PYTHONWARNINGS"] = "ignore"
        try:
            yield
        finally:
            logging.disable(previous_disable)
            if previous_env is None:
                del os.environ["PYTHONWARNINGS"]
            else:
                os.environ["PYTHONWARNINGS"] = previous_env
