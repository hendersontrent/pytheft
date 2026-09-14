import numpy as np
import pytest


def simulate(n_per_process: int = 4, length: int = 150, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate labelled time series from three processes, in the spirit of theft's simData."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, length + 1)
    generators = {
        "noise": lambda: rng.standard_normal(length),
        "sinusoid": lambda: np.sin(t) + rng.normal(0, 0.25, length),
        "random_walk": lambda: np.cumsum(rng.standard_normal(length)),
    }
    X = np.array([generate() for generate in generators.values() for _ in range(n_per_process)])
    y = np.repeat(list(generators), n_per_process)
    return X, y


@pytest.fixture
def simulated() -> tuple[np.ndarray, np.ndarray]:
    return simulate()


@pytest.fixture
def simulated_large() -> tuple[np.ndarray, np.ndarray]:
    return simulate(n_per_process=10)


@pytest.fixture
def unequal_series() -> dict[str, np.ndarray]:
    """Series of different lengths, with ids deliberately out of sorted order."""
    rng = np.random.default_rng(42)
    return {
        "b": rng.standard_normal(150),
        "a": np.cumsum(rng.standard_normal(130)),
        "c": np.sin(np.arange(140)) + rng.normal(0, 0.1, 140),
    }
