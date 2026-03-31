"""
conformal.py
------------
Split (inductive) conformal prediction for binary classification.
"""

import numpy as np


class SplitConformalClassifier:
    """
    Split conformal prediction for binary classification.

    After fitting on a calibration set, produces prediction sets
    C(x) ⊆ {0, 1} that satisfy marginal coverage P(y ∈ C(x)) ≥ 1 − alpha.

    Parameters
    ----------
    alpha : float
        Miscoverage level (default 0.10 → 90% coverage).
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha    = alpha
        self.q_hat    = None       # calibrated quantile threshold
        self._cal_scores = None    # stored for diagnostics

    # ── calibration ──────────────────────────────────────────────────────────

    def calibrate(self, probs: np.ndarray, y_cal: np.ndarray) -> None:
        """
        Compute nonconformity scores on calibration data and store quantile.

        Parameters
        ----------
        probs  : (n,) predicted probabilities P(Y=1|x) from base model.
        y_cal  : (n,) true binary labels.
        """
        probs = np.asarray(probs, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)

        # Nonconformity score: 1 - predicted probability of true class
        scores = np.where(y_cal == 1, 1.0 - probs, probs)
        self._cal_scores = scores

        n = len(scores)
        # Adjusted quantile for finite-sample guarantee
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self.q_hat = np.quantile(scores, level)

    # ── prediction ───────────────────────────────────────────────────────────

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        """
        Return prediction sets as boolean array of shape (n, 2).
        Column 0: whether label 0 is in the set.
        Column 1: whether label 1 is in the set.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        probs = np.asarray(probs, dtype=float)
        # Score for label 0: probs (= 1 - P(Y=1|x))  actually we use prob of class 0
        score_0 = probs          # nonconformity for y=0 is P(Y=1) (distance from 0)
        score_1 = 1.0 - probs   # nonconformity for y=1 is 1 - P(Y=1)
        in_0 = score_0 <= self.q_hat
        in_1 = score_1 <= self.q_hat
        return np.stack([in_0, in_1], axis=1)

    def predict_label(self, probs: np.ndarray) -> np.ndarray:
        """
        For point prediction: return the single label in the set when |C|=1,
        or -1 for empty sets, or 2 for both-label sets.
        """
        sets = self.predict_set(probs)
        sizes = sets.sum(axis=1)
        labels = np.where(sizes == 1,
                          np.argmax(sets.astype(int), axis=1),
                          np.where(sizes == 0, -1, 2))
        return labels

    # ── diagnostics ──────────────────────────────────────────────────────────

    def coverage(self, probs: np.ndarray, y_test: np.ndarray) -> float:
        """Empirical marginal coverage on test data."""
        sets   = self.predict_set(probs)
        y_test = np.asarray(y_test, dtype=int)
        covered = sets[np.arange(len(y_test)), y_test]
        return covered.mean()

    def avg_set_size(self, probs: np.ndarray) -> float:
        """Average prediction set size (efficiency measure)."""
        return self.predict_set(probs).sum(axis=1).mean()

    def coverage_violation(self, probs: np.ndarray, y_test: np.ndarray) -> float:
        """How far below nominal coverage (1-alpha) the empirical coverage falls."""
        return max(0.0, (1 - self.alpha) - self.coverage(probs, y_test))
