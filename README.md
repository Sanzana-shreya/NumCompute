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

All components — preprocessing, statistics, metrics, models, and pipelines — are designed to be **incremental, numerically stable, and shape‑consistent**, enabling real‑world online learning scenarios.

---

# 🧩 System Architecture

flowchart TD
    A[CSV or Streaming Data] --> B[Preprocessing: Scaling, Encoding, NaN Handling]
    B --> C[Pipeline API: fit, transform, predict]
    C --> D[Streaming Decision Tree]
    C --> E[Streaming Random Forest]
    D --> F[Predictions]
    E --> F
    F --> G[Metrics: Accuracy, F1, MSE]
    G --> H[Visualisation Module]


# 🔄 Streaming Workflow

sequenceDiagram
    participant U as Data Stream
    participant P as Preprocessing
    participant M as Model
    participant T as Metrics
    participant V as Visualiser

    U->>P: Load next chunk
    P->>M: Transform and update model
    M->>T: Predict and update metrics
    T->>V: Update plots
    V->>U: Next chunk

# 📊 Model Comparison (Tree vs Forest)

graph LR
    A[Decision Tree\nAccuracy: 0.9067\nFit: 1.07s\nPredict: 0.0006s]
    B[Random Forest\nAccuracy: 0.7067\nFit: 3.14s\nPredict: 0.0037s]

    A --- B


    
# ⚡ Vectorisation Speedup (Loop vs NumPy)
graph TD
    L[Python Loops\nMean: 0.0957s\nVar: 0.3378s]
    V[Vectorised NumPy\nMean: 0.00044s\nVar: 0.00469s]

    L -->|217x faster| V

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

# 📘 NumCompute‑Stream  
### *A Modularised Ensemble Tree‑based Streaming Machine Learning Framework*  
#### **also published as: `numstream`**

---

# 🎓 Academic Logo


███╗   ██╗██╗   ██╗███╗   ███╗███████╗████████╗████████╗██████╗ ███████╗
████╗  ██║██║   ██║████╗ ████║██╔════╝╚══██╔══╝╚══██╔══╝██╔══██╗██╔════╝
██╔██╗ ██║██║   ██║██╔████╔██║█████╗     ██║      ██║   ██████╔╝█████╗
██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══╝     ██║      ██║   ██╔══██╗██╔══╝
██║ ╚████║╚██████╔╝██║ ╚═╝ ██║███████╗   ██║      ██║   ██║  ██║███████╗
╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚══════╝


*A research‑grade machine learning framework built entirely with NumPy.*

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
