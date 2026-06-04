import numpy as np
import matplotlib.pyplot as plt


def plot_stream_metric(
    values,
    title: str = "Streaming Metric",
    ylabel: str = "Score",
    save_path: str | None = None,
    show: bool = True
):
    """
    Plot one metric over streaming chunks.

    Parameters
    ----------
    values : array-like
        Metric values over time.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        File path to save figure.
    show : bool
        Whether to display figure inline.
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(1, len(values) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(x, values, marker="o")
    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    histories: dict,
    title: str = "Model Comparison",
    ylabel: str = "Score",
    save_path: str | None = None,
    show: bool = True
):
    """
    Plot multiple model histories on one figure.

    Parameters
    ----------
    histories : dict
        Dictionary like {"model_name": [scores]}.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    save_path : str, optional
        File path to save figure.
    show : bool
        Whether to display figure inline.
    """
    plt.figure(figsize=(8, 4))

    for name, values in histories.items():
        values = np.asarray(values, dtype=float)
        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, marker="o", label=name)

    plt.title(title)
    plt.xlabel("Chunk")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()