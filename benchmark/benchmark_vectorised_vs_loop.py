import time
import numpy as np


def loop_mean(X):
    n_samples, n_features = X.shape
    out = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        total = 0.0
        for i in range(n_samples):
            total += X[i, j]
        out[j] = total / n_samples
    return out


def vectorised_mean(X):
    return np.mean(X, axis=0)


def loop_var(X):
    n_samples, n_features = X.shape
    mu = loop_mean(X)
    out = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        total = 0.0
        for i in range(n_samples):
            diff = X[i, j] - mu[j]
            total += diff * diff
        out[j] = total / n_samples
    return out


def vectorised_var(X):
    return np.var(X, axis=0)


def benchmark():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10000, 50))

    t0 = time.perf_counter()
    m_loop = loop_mean(X)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    m_vec = vectorised_mean(X)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    v_loop = loop_var(X)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    v_vec = vectorised_var(X)
    t7 = time.perf_counter()

    print("Mean close:", np.allclose(m_loop, m_vec))
    print("Variance close:", np.allclose(v_loop, v_vec))
    print(f"Loop mean time:       {t1 - t0:.6f}s")
    print(f"Vectorised mean time: {t3 - t2:.6f}s")
    print(f"Loop var time:        {t5 - t4:.6f}s")
    print(f"Vectorised var time:  {t7 - t6:.6f}s")


if __name__ == "__main__":
    benchmark()