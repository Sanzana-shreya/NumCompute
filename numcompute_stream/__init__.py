"""
NumCompute: A lightweight scientific computing toolkit built with Python and NumPy.

This package provides utilities for:
- CSV input/output
- preprocessing
- sorting and searching
- ranking and percentiles
- descriptive statistics
- evaluation metrics
- finite-difference optimisation helpers
- pipeline abstraction
- benchmarking support
"""

__version__ = "0.1.0"

from .tree import StreamingDecisionTreeClassifier
from .ensemble import StreamingRandomForestClassifier
from .stream import stream_chunks, StreamingMetricTracker
from .visualise import plot_stream_metric, plot_model_comparison