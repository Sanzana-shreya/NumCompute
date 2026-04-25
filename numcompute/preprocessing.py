import numpy as np

#scales numeric data so the mean becomes 0 and standard deviation becomes 1
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    #calculates mean and std from the data
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    #applies scaling using the stored mean and std
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
#scales data to a fixed range between 0 and 1
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    #finds minimum and maximum values from the data
    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self
    
    #scales values using stored min and max
    def transform(self, X):
        if self.min is None or self.max is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

#converts categorical values into one-hot encoder vectors
class OneHotEncoder:
    def __init__(self):
        self.categories = None

    #finds unique categories for each column
    def fit(self, X):
        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    #creates one-hot encoded columns based on learned categories
    def transform(self, X):
        if self.categories is None:
            raise ValueError("OneHotEncoder has not been fitted yet.")
        encoded_columns = []

        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories[i]

            one_hot = (col[:, None] == categories).astype(int)
            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

#applies different preprocessing to numeric and categorical columns
class ColumnTransformer:
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    #fits scaler on numeric columns and encoder on categorical columns
    def fit(self, X):
        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            self.scaler.fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.fit(X_cat)
        
        return self
    
    #transforms numeric and categorical parts separately and combines them
    def transform(self, X):
        if (len(self.num_cols) > 0 and self.scaler.mean is None) or \
           (len(self.cat_cols) > 0 and self.encoder.categories is None):
            raise ValueError("ColumnTransformer has not been fitted yet.")
        
        parts = []

        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            parts.append(self.scaler.transform(X_num))

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            parts.append(self.encoder.transform(X_cat))

        return np.hstack(parts)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
