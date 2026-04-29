# testing CSV reading functionality
import numpy as np
import pytest

from numcompute.io import read_csv


def test_read_csv_valid_file():
    # Path to dataset (update if needed in your system)
    file_path = r"C:\Users\aarib\OneDrive\Documents\AdelUni\Semester 1\ProgrammingAI\programming_task_1\data\Iris.csv"

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