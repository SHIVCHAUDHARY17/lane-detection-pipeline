import numpy as np


class LaneSmoother:
    """
    Stabilises lane detection across frames using exponential
    moving average of polynomial coefficients.

    Problem it solves:
        Dashed lane markings cause pixel dropout between dashes.
        When too few pixels exist, fit_polynomial returns None.
        Without smoothing this causes the lane to flicker or
        disappear for several frames.

    Solution:
        Keep a running weighted average of the last N valid fits.
        When current frame fails, use the remembered coefficients.
        When current frame succeeds, blend it with memory.

        This is the same principle used in Kalman filtering —
        combine prediction (memory) with measurement (current fit).

    Args:
        alpha: smoothing factor (0-1)
            higher = trust current frame more, less smoothing
            lower  = trust memory more, more smoothing
        max_age: how many frames to keep using memory after dropout
    """

    def __init__(self, alpha: float = 0.7, max_age: int = 10):
        self.alpha    = alpha
        self.max_age  = max_age

        # Stored smoothed coefficients for left and right lanes
        self._left_coeffs  = None
        self._right_coeffs = None

        # Frame counter since last valid detection
        self._left_age  = 0
        self._right_age = 0

    def update(self, left_coeffs, right_coeffs):
        """
        Update smoother with new polynomial coefficients.

        If new coefficients are valid:
            blend with memory using exponential moving average
        If new coefficients are None (dropout frame):
            keep using memory until max_age exceeded

        Args:
            left_coeffs: new left lane polynomial or None
            right_coeffs: new right lane polynomial or None

        Returns:
            smoothed left coefficients, smoothed right coefficients
        """
        self._left_coeffs  = self._smooth(
            self._left_coeffs, left_coeffs, "_left_age"
        )
        self._right_coeffs = self._smooth(
            self._right_coeffs, right_coeffs, "_right_age"
        )

        return self._left_coeffs, self._right_coeffs

    def _smooth(self, stored, current, age_attr: str):
        """
        Core smoothing logic for one lane.

        Exponential moving average:
            smoothed = alpha * current + (1 - alpha) * stored

        Args:
            stored: previously smoothed coefficients
            current: newly detected coefficients (can be None)
            age_attr: name of age counter attribute to update

        Returns:
            Updated smoothed coefficients
        """
        age = getattr(self, age_attr)

        if current is not None:
            # Valid detection — blend with memory
            setattr(self, age_attr, 0)
            if stored is None:
                # No memory yet — use current directly
                return current
            # Exponential moving average blend
            return self.alpha * current + (1 - self.alpha) * stored
        else:
            # Dropout frame — increment age counter
            age += 1
            setattr(self, age_attr, age)

            if age > self.max_age:
                # Memory too old — discard it
                return None

            # Return stored memory unchanged
            return stored