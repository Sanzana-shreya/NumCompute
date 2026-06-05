import time
import csv
from pathlib import Path

import numpy as np
import tracemalloc

from numcompute_stream.io import read_csv
from numcompute_stream.stats import mean, std
from numcompute_stream.sort_search import sort, topk
from numcompute_stream.rank import rank, percentile
from numcompute_stream.utils import pairwise_distances
from numcompute_stream.metrics import accuracy
from numcompute_stream import stream_chunks
from numcompute_stream.tree import StreamingDecisionTreeClassifier
from numcompute_stream.ensemble import StreamingRandomForestClassifier


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


def load_classification_csv(path):
    """
    Load dataset for streaming classification benchmark.

    Assumes:
    - first row is header
    - last column is target
    - non-numeric feature columns are ignored
    - target is label encoded using NumPy
    """
    raw = read_csv(path, delimiter=",", dtype=object)
    raw = np.asarray(raw, dtype=object)

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    data = raw[1:, :]

    if data.shape[1] < 2:
        raise ValueError(f"Dataset must have at least 2 columns: {path}")

    X_raw = data[:, :-1]
    y_raw = data[:, -1]

    numeric_cols = []

    for i in range(X_raw.shape[1]):
        col = X_raw[:, i]

        try:
            numeric_col = col.astype(float)
            numeric_cols.append(numeric_col)
        except (ValueError, TypeError):
            continue

    if not numeric_cols:
        raise ValueError(f"No numeric feature columns found in {path}")

    X = np.column_stack(numeric_cols)

    # remove bad rows
    valid_rows = np.isfinite(X).all(axis=1)
    X = X[valid_rows]
    y_raw = y_raw[valid_rows]

    classes, y = np.unique(y_raw, return_inverse=True)

    return X, y, classes


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


# ---------------- FUNCTION BENCHMARK ---------------- #

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

    # Pairwise distance benchmark
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


# ---------------- STREAMING MODEL BENCHMARK ---------------- #

def benchmark_streaming_model(model, X, y, classes, chunk_size=32):
    """
    Benchmark one streaming model.

    Returns
    -------
    dict
        Timing, memory, and final accuracy history.
    """
    tracemalloc.start()
    start = time.perf_counter()

    acc_history = []
    first_chunk = True

    for X_chunk, y_chunk in stream_chunks(X, y, chunk_size=chunk_size, shuffle=False):
        if not first_chunk:
            y_pred = model.predict(X_chunk)
            acc = accuracy(y_chunk, y_pred)
            acc_history.append(acc)

        if first_chunk:
            model.partial_fit(X_chunk, y_chunk, classes=np.arange(len(classes)))
            first_chunk = False
        else:
            model.partial_fit(X_chunk, y_chunk)

    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "time": end - start,
        "memory_mb": peak / (1024 ** 2),
        "final_accuracy": acc_history[-1] if len(acc_history) > 0 else None,
        "accuracy_history": acc_history,
    }


def benchmark_streaming_dataset(name, X, y, classes, chunk_size=32):
    """
    Benchmark decision tree and random forest on streaming data.
    """
    results = []

    tree_model = StreamingDecisionTreeClassifier(max_depth=5)
    rf_model = StreamingRandomForestClassifier(
        n_estimators=5,
        max_depth=5,
        random_state=42
    )

    tree_result = benchmark_streaming_model(
        tree_model, X, y, classes, chunk_size=chunk_size
    )

    rf_result = benchmark_streaming_model(
        rf_model, X, y, classes, chunk_size=chunk_size
    )

    results.append({
        "dataset": name,
        "model": "StreamingDecisionTreeClassifier",
        "time": tree_result["time"],
        "memory_mb": tree_result["memory_mb"],
        "final_accuracy": tree_result["final_accuracy"],
    })

    results.append({
        "dataset": name,
        "model": "StreamingRandomForestClassifier",
        "time": rf_result["time"],
        "memory_mb": rf_result["memory_mb"],
        "final_accuracy": rf_result["final_accuracy"],
    })

    return results


def save_results(results):
    out_path = ROOT / "benchmark_results.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to: {out_path}")


def save_streaming_results(results):
    out_path = ROOT / "streaming_benchmark_results.csv"

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


def print_streaming_results(results):
    print("\nSTREAMING MODEL BENCHMARK RESULTS")
    print("-" * 90)

    print(
        f"{'Dataset':<12}"
        f"{'Model':<35}"
        f"{'Time(s)':<12}"
        f"{'Memory(MB)':<15}"
        f"{'Final Accuracy'}"
    )

    print("-" * 90)

    for r in results:
        final_acc = r["final_accuracy"]
        final_acc_str = f"{final_acc:.4f}" if final_acc is not None else "N/A"

        print(
            f"{r['dataset']:<12}"
            f"{r['model']:<35}"
            f"{r['time']:<12.6f}"
            f"{r['memory_mb']:<15.4f}"
            f"{final_acc_str}"
        )


def run_benchmarks(datasets, save_csv=True):
    all_results = []

    print("Loading datasets...\n")

    for name, path in datasets.items():
        path = Path(path)

        if not path.exists():
            print(f"{name} not found -> {path}")
            continue

        X = load_numeric_csv(path)
        print(f"{name}: shape={X.shape}")

        all_results.extend(benchmark_dataset(name, X))

    print_results(all_results)

    if save_csv and len(all_results) > 0:
        save_results(all_results)

    return all_results


def run_streaming_benchmarks(datasets, save_csv=True, chunk_size=32):
    all_results = []

    print("Loading datasets for streaming benchmarks...\n")

    for name, path in datasets.items():
        path = Path(path)

        if not path.exists():
            print(f"{name} not found -> {path}")
            continue

        try:
            X, y, classes = load_classification_csv(path)
            print(f"{name}: X shape={X.shape}, y shape={y.shape}")

            results = benchmark_streaming_dataset(
                name, X, y, classes, chunk_size=chunk_size
            )
            all_results.extend(results)

        except Exception as e:
            print(f"{name} skipped: {e}")

    if len(all_results) > 0:
        print_streaming_results(all_results)

        if save_csv:
            save_streaming_results(all_results)

    return all_results


def main():
    run_benchmarks(DATASETS, save_csv=True)
    run_streaming_benchmarks(DATASETS, save_csv=True, chunk_size=32)


if __name__ == "__main__":
    main()