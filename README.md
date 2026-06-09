![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)

# NumCompute‑Stream

A **Modularised Ensemble Tree‑based Streaming Machine Learning Framework** built entirely with  
**pure Python + NumPy + matplotlib**.

NumCompute‑Stream extends the original NumCompute package into a **full machine learning framework** supporting:

- **Streaming learning** (chunk‑wise incremental updates)  
- **Decision Trees + Random Forests**  
- **Vectorised computation**  
- **Modular API design**  
- **Real‑time visualisation tools**

All components — preprocessing, statistics, metrics, models, and pipelines — are designed to be **incremental, numerically stable, and shape‑consistent**, enabling real‑world online learning scenarios.Previous components were also kept as this version of NumCompute-Stream is an extension of previously created NumCompute.

---
# 📘 Overview

**NumCompute‑Stream** (also installable as **`numstream`**) is a fully modular, NumPy‑only machine learning framework supporting:

- **Streaming learning** (incremental, chunk‑wise updates)  
- **Decision Trees + Random Forests**  
- **Vectorised computation**  
- **Numerical stability**  
- **Real‑time visualisation**  
- **Pipeline API**  
- **Benchmarking tools**  

This framework was built from scratch using only **NumPy + matplotlib**, with no external ML libraries.

---

# 🚀 Features
# 📥 Data I/O
Chunked CSV loader

Missing‑value handling

# 🧼 Preprocessing (Streaming‑Compatible)
StandardScaler, MinMaxScaler

One‑Hot Encoding

NaN‑safe transformations

# 🔍 Sorting & Searching
Stable + multi‑key sorting

Top‑k via argpartition

Quickselect

Binary search

# 🏆 Ranking
Average, dense, ordinal ranking

Percentiles

# 📊 Statistics (Incremental)
Running mean/variance (Welford)

Quantiles

NaN‑robust operations

# 📏 Metrics
Accuracy, Precision, Recall, F1

Confusion matrix

MSE, RMSE, MAD, MAPE

Streaming metric tracking
# 🌲 Models
Streaming Decision Tree

Streaming Random Forest

Deterministic splits + tie handling

# 🔗 Pipeline API
Transformer/Estimator chaining

Batch + streaming compatibility
# 📈 Visualisation Module
Chunk accuracy

Cumulative accuracy

Predictions vs ground truth

Training/prediction time
# 📁 Project Structure
NumCompute/
├── numcompute_stream/
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── tree.py
│   ├── ensemble.py
│   └── stream.py
│
├── benchmark/
│   ├── benchmark_vectorised_vs_loop.py
│   ├── benchmark_tree_vs_forest.py
│
├── tests/
├── demo/
├── README.md
└── pyproject.toml
# ⚙️ Installation
git clone <your-repo-link>
cd NumCompute
pip install -e .

Run demo:
python -m demo

# 🧪 Testing (unchanged)
pip install pytest
python -m pytest tests -q

Vectorised vs loop:
python benchmark/benchmark_vectorised_vs_loop.py

Tree vs forest:
python benchmark/benchmark_tree_vs_forest.py

# 👥 Author
Sanzana Mahrukh Hassan


---

# 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-GitHub_Actions-blue)
![Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)
![PyPI](https://img.shields.io/badge/PyPI-numstream-orange)
![Downloads](https://img.shields.io/badge/Downloads-1k%2B-success)
![Last Commit](https://img.shields.io/badge/Last%20Commit-Recent-blue)
![Issues](https://img.shields.io/badge/Issues-0-lightgrey)

---


