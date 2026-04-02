import numpy as np
import pytest
from src.smoother import LaneSmoother


def make_coeffs(a=0.0, b=0.0, c=300.0):
    """Create simple polynomial coefficients for testing."""
    return np.array([a, b, c])


def test_smoother_returns_current_when_no_memory():
    """
    First frame with valid detection should return current coefficients.
    """
    smoother = LaneSmoother(alpha=0.7, max_age=5)
    coeffs   = make_coeffs(c=300.0)

    left_out, _ = smoother.update(coeffs, None)

    assert left_out is not None, "Should return coefficients on first valid frame"
    np.testing.assert_array_almost_equal(left_out, coeffs)


def test_smoother_holds_memory_during_dropout():
    """
    When detection drops out, smoother should return last valid coefficients.
    """
    smoother = LaneSmoother(alpha=0.7, max_age=5)
    coeffs   = make_coeffs(c=300.0)

    # Feed one valid frame
    smoother.update(coeffs, None)

    # Now feed None (dropout) — should still return something
    left_out, _ = smoother.update(None, None)

    assert left_out is not None, \
        "Smoother should hold memory during dropout frames"


def test_smoother_forgets_after_max_age():
    """
    After max_age consecutive dropout frames, memory should be cleared.
    """
    smoother = LaneSmoother(alpha=0.7, max_age=3)
    coeffs   = make_coeffs(c=300.0)

    # Feed one valid frame
    smoother.update(coeffs, None)

    # Feed max_age + 1 dropout frames
    for _ in range(4):
        left_out, _ = smoother.update(None, None)

    assert left_out is None, \
        "Smoother should forget after exceeding max_age"


def test_smoother_blends_coefficients():
    """
    With alpha=1.0 (trust current fully), output should equal current.
    """
    smoother = LaneSmoother(alpha=1.0, max_age=5)
    first    = make_coeffs(c=300.0)
    second   = make_coeffs(c=400.0)

    smoother.update(first, None)
    left_out, _ = smoother.update(second, None)

    np.testing.assert_array_almost_equal(left_out, second), \
        "With alpha=1.0 output should exactly equal current coefficients"