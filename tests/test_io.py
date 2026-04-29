import os
from numcompute.io import read_csv

# Project root (assuming notebook is inside /demo)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Construct dataset path
file_path = os.path.join(BASE_DIR, "data", "Iris.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

# Load dataset
data = read_csv(file_path)

print(data)