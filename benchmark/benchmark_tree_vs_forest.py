import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from numcompute_stream.tree import StreamingDecisionTreeClassifier
from numcompute_stream.ensemble import StreamingRandomForestClassifier


def make_synthetic_stream(
    n_samples=1000,
    n_features=10,
    chunk_size=100,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    X = rng.normal(size=(n_samples, n_features))

    y = (
        X[:, 0]
        + 0.8 * X[:, 1]
        - 0.5 * X[:, 2]
        > 0
    ).astype(int)

    chunks = []

    for i in range(0, n_samples, chunk_size):
        chunks.append(
            (
                X[i:i + chunk_size],
                y[i:i + chunk_size]
            )
        )

    return chunks


def prequential_benchmark(model, chunks):
    fit_times = []
    pred_times = []
    accuracies = []

    first_chunk = True

    for i, (X_chunk, y_chunk) in enumerate(chunks):

        print(f"Processing chunk {i + 1}/{len(chunks)}")

        if first_chunk:

            start = time.perf_counter()

            model.partial_fit(
                X_chunk,
                y_chunk,
                classes=np.array([0, 1])
            )

            end = time.perf_counter()

            fit_times.append(end - start)
            pred_times.append(np.nan)
            accuracies.append(np.nan)

            first_chunk = False
            continue

        start = time.perf_counter()

        y_pred = model.predict(X_chunk)

        end = time.perf_counter()

        pred_times.append(end - start)

        acc = np.mean(y_pred == y_chunk)
        accuracies.append(acc)

        start = time.perf_counter()

        model.partial_fit(X_chunk, y_chunk)

        end = time.perf_counter()

        fit_times.append(end - start)

    return {
        "fit_times": np.array(fit_times),
        "pred_times": np.array(pred_times),
        "accuracies": np.array(accuracies),
        "mean_fit_time": np.nanmean(fit_times),
        "mean_pred_time": np.nanmean(pred_times),
        "mean_accuracy": np.nanmean(accuracies),
    }


def main():

    print("Creating stream...")

    chunks = make_synthetic_stream(
        n_samples=1000,
        n_features=10,
        chunk_size=100
    )

    print("Creating Decision Tree...")

    tree = StreamingDecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        criterion="gini"
    )

    print("Running Decision Tree benchmark...")

    tree_results = prequential_benchmark(
        tree,
        chunks
    )

    print("Creating Random Forest...")

    forest = StreamingRandomForestClassifier(
        n_estimators=5,
        max_depth=5,
        min_samples_split=4,
        criterion="gini",
        bootstrap=True,
        random_state=42
    )

    print("Running Random Forest benchmark...")

    forest_results = prequential_benchmark(
        forest,
        chunks
    )

    print("\n===== Decision Tree =====")
    print("Mean Fit Time:", tree_results["mean_fit_time"])
    print("Mean Predict Time:", tree_results["mean_pred_time"])
    print("Mean Accuracy:", tree_results["mean_accuracy"])

    print("\n===== Random Forest =====")
    print("Mean Fit Time:", forest_results["mean_fit_time"])
    print("Mean Predict Time:", forest_results["mean_pred_time"])
    print("Mean Accuracy:", forest_results["mean_accuracy"])


if __name__ == "__main__":
    main()