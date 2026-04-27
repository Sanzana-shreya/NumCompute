import numpy as np

#reads a CSV file and converts it into a numpy array
def read_csv(file_path, delimiter=',', dtype=object,missing_val=''):
    
    #using genfromtxt so it can handle missing values
    arr = np.genfromtxt(file_path, delimiter=delimiter,dtype=dtype, missing_values=missing_val, filling_values=np.nan,encoding="utf-8")
    return np.vectorize(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
    )(arr)
