import time
import csv
from pathlib import Path

import numpy as np
import tracemalloc

from numcompute.io import read_csv
from numcompute.stats import mean, std
from numcompute.sort_search import sort, topk
from numcompute.rank import rank, percentile
from numcompute.utils import pairwise_distances


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

DATASETS = {
    "WineQT": DATA_DIR / "WineQT.csv",
    "Iris": DATA_DIR / "Iris.csv",
    "Sleep": DATA_DIR / "sleep_data.csv",
}


def timer(func, *args, repeat=5):
    times = []

    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return min(times)

def measure_memory(func, *args):
    tracemalloc.start()

    func(*args)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / (1024 ** 2)  # Convert to MB


def load_numeric_csv(path):
    raw = read_csv(path, delimiter=",", dtype=object)

    # If CSV has header row, remove it
    raw = np.asarray(raw, dtype=object)

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    # Remove first row as header
    data = raw[1:, :]

    numeric_cols = []

    for i in range(data.shape[1]):
        col = data[:, i]

        try:
            numeric_col = col.astype(float)
            numeric_cols.append(numeric_col)
        except (ValueError, TypeError):
            # Categorical columns are ignored for benchmarking
            continue

    if not numeric_cols:
        raise ValueError(f"No numeric columns found in {path}")

    X = np.column_stack(numeric_cols)

    # Remove the rows with NaN or infinity
    X = X[np.isfinite(X).all(axis=1)]

    return X


# ---------------- LOOP BASELINES ---------------- #

def loop_mean(arr):
    total = 0.0
    count = 0

    for v in arr:
        if not np.isnan(v):
            total += v
            count += 1

    return total / count


def loop_std(arr):
    m = loop_mean(arr)
    total = 0.0
    count = 0

    for v in arr:
        if not np.isnan(v):
            total += (v - m) ** 2
            count += 1

    return (total / count) ** 0.5


def loop_sort(arr):
    return sorted(arr)


def loop_topk(arr, k):
    return sorted(arr)[-k:]


def loop_rank(arr):
    sorted_vals = sorted(arr)
    result = []

    for v in arr:
        positions = [i + 1 for i, x in enumerate(sorted_vals) if x == v]
        result.append(sum(positions) / len(positions))

    return np.array(result)


def loop_percentile(arr, q):
    arr = sorted(arr)
    pos = (q / 100) * (len(arr) - 1)

    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if lower == upper:
        return arr[lower]

    weight = pos - lower
    return arr[lower] * (1 - weight) + arr[upper] * weight


def loop_pairwise_distances(X, Y):
    distances = []

    for x in X:
        row = []
        for y in Y:
            dist = np.sqrt(np.sum((x - y) ** 2))
            row.append(dist)
        distances.append(row)

    return np.array(distances)


# ---------------- BENCHMARK ---------------- #

def benchmark_dataset(name, X):
    results = []

    flat = X.ravel()
    flat = flat[np.isfinite(flat)]

    if len(flat) > 5000:
        flat = flat[:5000]

    k = min(10, len(flat))

    tests = [
        ("mean", loop_mean, mean, np.nanmean, (flat,)),
        ("std", loop_std, std, np.nanstd, (flat,)),
        ("sort", loop_sort, sort, np.sort, (flat,)),
        ("topk", loop_topk, topk, None, (flat, k)),
        ("rank", loop_rank, rank, None, (flat,)),
        ("percentile", loop_percentile, percentile, np.nanpercentile, (flat, 50)),
    ]

    # Loop through all the operations and benchmark each one
    for op, loop_f, my_f, np_f, args in tests:
        t_loop = timer(loop_f, *args)
        m_loop = measure_memory(loop_f, *args)

        t_my = timer(my_f, *args)
        m_my = measure_memory(my_f, *args)

        t_np = timer(np_f, *args) if np_f else None
        m_np = measure_memory(np_f, *args) if np_f else None

        results.append({
            "dataset": name,
            "operation": op,
            "loop_time": t_loop,
            "numcompute_time": t_my,
            "numpy_time": t_np,
            "loop_memory": m_loop,
            "numcompute_memory": m_my,
            "numpy_memory": m_np,
            "speedup": t_loop / t_my if t_my > 0 else np.inf,
        })

    # Pairwise distance benchmark from utils.py
    # Kept separate because pairwise distance is O(n^2)
    n_pairwise = min(100, X.shape[0])
    X_small = X[:n_pairwise, :]

    t_loop_pairwise = timer(loop_pairwise_distances, X_small, X_small)
    m_loop_pairwise = measure_memory(loop_pairwise_distances, X_small, X_small)

    t_my_pairwise = timer(pairwise_distances, X_small, X_small)
    m_my_pairwise = measure_memory(pairwise_distances, X_small, X_small)

    results.append({
        "dataset": name,
        "operation": "pairwise_distances",
        "loop_time": t_loop_pairwise,
        "numcompute_time": t_my_pairwise,
        "numpy_time": None,
        "loop_memory": m_loop_pairwise,
        "numcompute_memory": m_my_pairwise,
        "numpy_memory": None,
        "speedup": t_loop_pairwise / t_my_pairwise if t_my_pairwise > 0 else np.inf,
    })

    return results


def save_results(results):
    out_path = ROOT / "benchmark_results.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {out_path}")

def print_results(results):
    print("\nBENCHMARK RESULTS")
    print("-" * 120)

    print(
        f"{'Dataset':<10}"
        f"{'Operation':<20}"
        f"{'Loop(s)':<10}"
        f"{'NumComp(s)':<12}"
        f"{'Loop(MB)':<10}"
        f"{'NumComp(MB)':<12}"
        f"{'Speedup'}"
    )

    print("-" * 120)

    for r in results:
        print(
            f"{r['dataset']:<10}"
            f"{r['operation']:<20}"
            f"{r['loop_time']:<10.6f}"
            f"{r['numcompute_time']:<12.6f}"
            f"{r['loop_memory']:<10.4f}"
            f"{r['numcompute_memory']:<12.4f}"
            f"{r['speedup']:.2f}x"
        )



def run_benchmarks(datasets, save_csv=True):
    all_results = []

    print("Loading datasets...\n")

    for name, path in datasets.items():
        path = Path(path)

        if not path.exists():
            print(f"{name} not found → {path}")
            continue

        X = load_numeric_csv(path)
        print(f"{name}: shape={X.shape}")

        all_results.extend(benchmark_dataset(name, X))

    print_results(all_results)

    if save_csv:
        save_results(all_results)

    return all_results


def main():
    run_benchmarks(DATASETS, save_csv=True)


if __name__ == "__main__":
    main()