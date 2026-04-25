import numpy as np

#reads a CSV file and converts it into a numpy array
def read_csv(file_path, delimiter=',', dtype=float,missing_val=''):
    
    #using genfromtxt so it can handle missing values
    return np.genfromtxt(file_path, delimiter=delimiter,dtype=dtype, missing_values=missing_val, filling_values=np.nan)

