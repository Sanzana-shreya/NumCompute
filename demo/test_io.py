import numpy as np
from numcompute.io import read_csv

file_path = r"C:\Users\aarib\OneDrive\Documents\AdelUni\Semester 1\ProgrammingAI\programming_task_1\data\Iris.csv"
data = read_csv(file_path)
print(data)