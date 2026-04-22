import numpy as np

def read_csv(file_path, delimiter=',', dtype=float,missing_val=''):
    
    return np.genfromtxt(file_path, delimiter=delimiter,dtype=dtype, missing_values=missing_val, filling_values=np.nan)

