# testing CSV reading functionality
from pathlib import Path
import numpy as np
import pytest

from numcompute_stream.io import read_csv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def test_read_csv_valid_file():
    # Path to dataset
    file_path = DATA_DIR / "Iris.csv"

    # Read CSV file
    data = read_csv(file_path)

    # Check data is not None
    assert data is not None

    # Check it returns a NumPy array
    assert isinstance(data, np.ndarray)

    # Check it has rows
    assert data.shape[0] > 0


def test_read_csv_invalid_path():
    # Non-existent file should raise error
    file_path = "invalid_path.csv"

    with pytest.raises(Exception):
        read_csv(file_path)