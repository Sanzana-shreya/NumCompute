"""
Visualization utilities for streaming machine learning experiments.

Provides:
- Streaming metric plotting
- Multi-model comparison plotting
- Prediction vs ground truth visualization
- Backward compatibility for older notebook APIs
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# SINGLE METRIC PLOTTING
# =========================================================
def plot_stream_metric(
    values,
    title: str = "Streaming Metric",
    ylabel: str = "Score",
    save_path: str | None = None,
    show: bool = True
):
    """Plot a metric over streaming chunks."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Empty metric list provided")

    x = np.arange(1, len(values) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker="o", linewidth=2)

    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# MULTI-MODEL COMPARISON
# =========================================================
def plot_model_comparison(
    histories: dict,
    title: str = "Model Comparison",
    ylabel: str = "Score",
    save_path: str | None = None,
    show: bool = True
):
    """Compare multiple models over time."""
    plt.figure(figsize=(8, 4))

    for name, values in histories.items():
        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            continue

        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, marker="o", linewidth=2, label=name)

    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# PREDICTION VS GROUND TRUTH (FIXED & STABLE)
# =========================================================
def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "Ground Truth",
    ylabel: str = "Predictions",
    save_path: str | None = None,
    show: bool = True
):
    """Scatter plot comparing predictions vs actual values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty input provided")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    plt.figure(figsize=(6, 6))

    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Ideal line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# COMPATIBILITY LAYER
# =========================================================
def plot_metric_over_time(values, title="Streaming Metric", ylabel="Score", **kwargs):
    return plot_stream_metric(values, title=title, ylabel=ylabel, **kwargs)


def compare_models(histories, title="Model Comparison", ylabel="Score", **kwargs):
    return plot_model_comparison(histories, title=title, ylabel=ylabel, **kwargs)