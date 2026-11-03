import numpy as np

def _validate_values(final_values: np.ndarray) -> np.ndarray:
    values = np.asarray(final_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("final_values must be a non-empty, finite, one-dimensional array")
    return values


def summarise(final_values: np.ndarray) -> dict[str, float]:
    final_values = _validate_values(final_values)
    return {
        "median": float(np.median(final_values)),
        "mean": float(np.mean(final_values)),
        "p10": float(np.percentile(final_values, 10)),
        "p90": float(np.percentile(final_values, 90)),
        "min": float(np.min(final_values)),
        "max": float(np.max(final_values)),
    }

def prob_reach_goal(final_values: np.ndarray, goal: float) -> float:
    """Fraction meeting a positive real-terms goal; non-positive goals disable it."""
    final_values = _validate_values(final_values)
    if not np.isfinite(goal):
        raise ValueError("goal must be finite")
    if goal <= 0:
        return 0.0
    return float((final_values >= goal).mean())
