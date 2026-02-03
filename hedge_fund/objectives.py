"""
Custom XGBoost objective functions for institutional-grade model optimization.

Provides loss functions that directly optimize for trading metrics
(profit factor, directional accuracy) rather than generic regression loss.
"""

import numpy as np


def asymmetric_loss_objective(y_true, y_pred):
    """
    Asymmetric loss for high win-rate optimization.

    Penalizes wrong-direction predictions 10x more than missed opportunities,
    forcing the model to only predict when highly confident.

    Args:
        y_true: Array of true R-values.
        y_pred: Array of predicted R-values.

    Returns:
        Tuple of (gradient, hessian) arrays for XGBoost custom objective.
    """
    residual = y_true - y_pred
    wrong_direction = np.sign(y_true) != np.sign(y_pred)

    penalty = np.where(wrong_direction, 10.0, 1.0)

    grad = -2.0 * penalty * residual
    hess = 2.0 * penalty

    return grad, hess


def profit_factor_objective(y_true, y_pred):
    """
    Differentiable profit factor objective for XGBoost.

    Directly optimizes Profit Factor = Gross_Wins / Gross_Losses using
    soft sigmoid classification for differentiability.

    Mathematical strategy:
      1. Classify predictions as winners/losers using soft sigmoid on
         the agreement score (y_true * y_pred).
      2. Compute gross profit and gross loss.
      3. Minimize -log(PF).

    Args:
        y_true: Array of true R-values.
        y_pred: Array of predicted R-values.

    Returns:
        Tuple of (gradient, hessian) arrays for XGBoost custom objective.
    """
    epsilon = 1e-6

    agreement = y_true * y_pred
    temp = 2.0
    soft_win_prob = 1.0 / (1.0 + np.exp(-agreement / temp))
    soft_loss_prob = 1.0 - soft_win_prob

    win_contribution = soft_win_prob * np.abs(y_true)
    loss_contribution = soft_loss_prob * np.abs(y_true)

    gross_profit = np.sum(win_contribution) + epsilon
    gross_loss = np.sum(loss_contribution) + epsilon

    sigmoid_derivative = soft_win_prob * (1.0 - soft_win_prob)
    d_win_prob = sigmoid_derivative * y_true / temp
    d_loss_prob = -d_win_prob

    grad = (
        -(1.0 / gross_profit) * d_win_prob * np.abs(y_true)
        + (1.0 / gross_loss) * d_loss_prob * np.abs(y_true)
    )

    hess = sigmoid_derivative * (1.0 / (temp**2)) * np.abs(y_true) + epsilon

    return grad, hess
