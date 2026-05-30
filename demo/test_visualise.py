import os
import tempfile

from numcompute.visualise import plot_stream_metric, plot_model_comparison


# For visualise.py


def test_plot_stream_metric_save():
    # Save single metric plot
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "metric.png")
        plot_stream_metric([0.5, 0.6, 0.7], save_path=path, show=False)

        assert os.path.exists(path)


def test_plot_model_comparison_save():
    # Save comparison plot
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "comparison.png")
        plot_model_comparison(
            {"tree": [0.5, 0.6], "rf": [0.6, 0.7]},
            save_path=path,
            show=False
        )

        assert os.path.exists(path)