import numpy as np
import warnings


class Imputer:
    """
    Replace missing numeric values using mean, median, or constant strategy.

    Parameters
    ----------
    strategy : str, default="mean"
        Imputation strategy. Supported values are "mean", "median", and "constant".
    fill_value : float, default=0
        Value used when strategy="constant" or when a column contains only NaN values.

    Attributes
    ----------
    statistics : ndarray of shape (n_features,)
        Learned replacement value for each column.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(total_seen_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self, strategy="mean", fill_value=0):
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError("Invalid strategy. Supported strategies are: 'mean', 'median', 'constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics = None
        self._X_seen = None

    def fit(self, X):
        """
        Compute replacement statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : Imputer

        Raises
        ------
        ValueError
            If X is not a 2D array.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Imputer expects a 2D array")

        # Suppress RuntimeWarning from all-NaN columns (correct way)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            if self.strategy == "mean":
                self.statistics = np.nanmean(X, axis=0)

            elif self.strategy == "median":
                self.statistics = np.nanmedian(X, axis=0)

            elif self.strategy == "constant":
                self.statistics = np.full(X.shape[1], self.fill_value)

        # Replace NaN statistics (from all-NaN columns) with fill_value
        self.statistics = np.where(
            np.isnan(self.statistics),
            self.fill_value,
            self.statistics
        )

        self._X_seen = X.copy()

        return self

    def partial_fit(self, X):
        """
        Incrementally update replacement statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : Imputer
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Imputer expects a 2D array")

        if self._X_seen is None:
            self._X_seen = X.copy()
        else:
            if X.shape[1] != self._X_seen.shape[1]:
                raise ValueError("X has different number of columns than fitted data")

            self._X_seen = np.vstack([self._X_seen, X])

        return self.fit(self._X_seen)

    def transform(self, X):
        """
        Replace NaN values using learned statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)

        Raises
        ------
        ValueError
            If the imputer has not been fitted, if X is not 2D, or if column count differs.
        """
        if self.statistics is None:
            raise ValueError("Imputer has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Imputer expects a 2D array")

        if X.shape[1] != len(self.statistics):
            raise ValueError("X has different number of columns than fitted data")

        X_out = X.copy()

        # Replace only missing positions
        nan_rows, nan_cols = np.where(np.isnan(X_out))
        X_out[nan_rows, nan_cols] = self.statistics[nan_cols]

        return X_out

    def fit_transform(self, X):
        """
        Fit the imputer and transform X in one step.
        """
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Standardize numeric features using z-score scaling.

    Formula
    -------
    X_scaled = (X - mean) / std

    Attributes
    ----------
    mean : ndarray of shape (n_features,)
    std : ndarray of shape (n_features,)

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(total_seen_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self._X_seen = None

    def fit(self, X):
        """
        Compute column-wise mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array")

        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)

        # Avoid division by zero for constant columns
        self.std = np.where(self.std == 0, 1, self.std)

        self._X_seen = X.copy()

        return self

    def partial_fit(self, X):
        """
        Incrementally update mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array")

        if self._X_seen is None:
            self._X_seen = X.copy()
        else:
            if X.shape[1] != self._X_seen.shape[1]:
                raise ValueError("X has different number of columns than fitted data")

            self._X_seen = np.vstack([self._X_seen, X])

        return self.fit(self._X_seen)

    def transform(self, X):
        """
        Apply z-score scaling using learned mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array")

        if X.shape[1] != len(self.mean):
            raise ValueError("X has different number of columns than fitted data")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """
        Fit the scaler and transform X in one step.
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Scale numeric features to a fixed range.

    Formula
    -------
    X_scaled = ((X - min) / (max - min)) * (high - low) + low

    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired output range.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(total_seen_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self, feature_range=(0, 1)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must contain two values")

        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range minimum must be less than maximum")

        self.feature_range = feature_range
        self.min = None
        self.max = None
        self.range = None
        self._X_seen = None

    def fit(self, X):
        """
        Compute column-wise minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array")

        self.min = np.nanmin(X, axis=0)
        self.max = np.nanmax(X, axis=0)
        self.range = self.max - self.min

        # Avoid division by zero for constant columns
        self.range = np.where(self.range == 0, 1, self.range)

        self._X_seen = X.copy()

        return self

    def partial_fit(self, X):
        """
        Incrementally update minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array")

        if self._X_seen is None:
            self._X_seen = X.copy()
        else:
            if X.shape[1] != self._X_seen.shape[1]:
                raise ValueError("X has different number of columns than fitted data")

            self._X_seen = np.vstack([self._X_seen, X])

        return self.fit(self._X_seen)

    def transform(self, X):
        """
        Apply min-max scaling using learned minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        if self.min is None or self.max is None or self.range is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array")

        if X.shape[1] != len(self.min):
            raise ValueError("X has different number of columns than fitted data")

        low, high = self.feature_range
        X_scaled = (X - self.min) / self.range

        return X_scaled * (high - low) + low

    def fit_transform(self, X):
        """
        Fit the scaler and transform X in one step.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Encode categorical columns as one-hot numeric arrays.

    Attributes
    ----------
    categories : list of ndarray
        Unique categories learned for each column.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features log n_samples)
    partial_fit: O(total_seen_samples * n_features log total_seen_samples)
    transform: O(n_samples * total_categories)
    """

    def __init__(self):
        self.categories = None
        self._X_seen = None

    def fit(self, X):
        """
        Learn unique categories for each categorical column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : OneHotEncoder
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 2D array")

        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self._X_seen = X.copy()

        return self

    def partial_fit(self, X):
        """
        Incrementally update learned categories.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : OneHotEncoder
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 2D array")

        if self._X_seen is None:
            self._X_seen = X.copy()
        else:
            if X.shape[1] != self._X_seen.shape[1]:
                raise ValueError("X has different number of columns than fitted data")

            self._X_seen = np.vstack([self._X_seen, X])

        return self.fit(self._X_seen)

    def transform(self, X):
        """
        Transform categorical values into one-hot encoded columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_encoded : ndarray of shape (n_samples, total_encoded_features)
        """
        if self.categories is None:
            raise ValueError("OneHotEncoder has not been fitted yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 2D array")

        if X.shape[1] != len(self.categories):
            raise ValueError("X has different number of columns than fitted data")

        encoded_columns = []

        # Loop is used because each categorical column may have different categories
        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories[i]

            one_hot = (col[:, None] == categories).astype(int)
            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)

    def fit_transform(self, X):
        """
        Fit the encoder and transform X in one step.
        """
        return self.fit(X).transform(X)


class ColumnTransformer:
    """
    Apply separate preprocessing to numeric and categorical columns.

    Numeric columns:
        Imputer -> StandardScaler

    Categorical columns:
        OneHotEncoder

    If num_cols and cat_cols are not provided, column types are automatically detected:
        - Columns convertible to float are treated as numeric
        - Remaining columns are treated as categorical

    Parameters
    ----------
    num_cols : list of int, optional
        Indices of numeric columns.
    cat_cols : list of int, optional
        Indices of categorical columns.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(total_seen_samples * n_features)
    transform: O(n_samples * transformed_features)
    """

    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

        self.imputer = Imputer(strategy="mean")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def _detect_columns(self, X):
        """
        Automatically detect numeric and categorical columns.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        num_cols : list of int
        cat_cols : list of int
        """
        num_cols = []
        cat_cols = []

        for i in range(X.shape[1]):
            col = X[:, i]

            try:
                col.astype(float)
                num_cols.append(i)
            except (ValueError, TypeError):
                cat_cols.append(i)

        return num_cols, cat_cols

    def fit(self, X):
        """
        Fit numeric and categorical preprocessing components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : ColumnTransformer
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("ColumnTransformer expects a 2D array")

        # Auto-detect columns if not provided
        if self.num_cols is None and self.cat_cols is None:
            self.num_cols, self.cat_cols = self._detect_columns(X)

        self.num_cols = [] if self.num_cols is None else self.num_cols
        self.cat_cols = [] if self.cat_cols is None else self.cat_cols

        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            X_num = self.imputer.fit_transform(X_num)
            self.scaler.fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.fit(X_cat)

        return self

    def partial_fit(self, X, y=None):
        """
        Incrementally update numeric and categorical preprocessing components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored
            Included for pipeline compatibility.

        Returns
        -------
        self : ColumnTransformer
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("ColumnTransformer expects a 2D array")

        # Auto-detect columns if not provided
        if self.num_cols is None and self.cat_cols is None:
            self.num_cols, self.cat_cols = self._detect_columns(X)

        self.num_cols = [] if self.num_cols is None else self.num_cols
        self.cat_cols = [] if self.cat_cols is None else self.cat_cols

        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            self.imputer.partial_fit(X_num)
            X_num = self.imputer.transform(X_num)
            self.scaler.partial_fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.partial_fit(X_cat)

        return self

    def transform(self, X):
        """
        Transform selected numeric and categorical columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of shape (n_samples, transformed_features)
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("ColumnTransformer expects a 2D array")

        parts = []

        if len(self.num_cols) > 0:
            if self.scaler.mean is None:
                raise ValueError("ColumnTransformer numeric part has not been fitted yet.")

            X_num = X[:, self.num_cols].astype(float)
            X_num = self.imputer.transform(X_num)
            X_num = self.scaler.transform(X_num)
            parts.append(X_num)

        if len(self.cat_cols) > 0:
            if self.encoder.categories is None:
                raise ValueError("ColumnTransformer categorical part has not been fitted yet.")

            X_cat = X[:, self.cat_cols]
            X_cat = self.encoder.transform(X_cat)
            parts.append(X_cat)

        if not parts:
            raise ValueError("No columns were selected for transformation")

        return np.hstack(parts)

    def fit_transform(self, X):
        """
        Fit all preprocessing components and transform X in one step.
        """
        return self.fit(X).transform(X)