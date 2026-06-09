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

```mermaid
flowchart TD
    A[CSV / Streaming Data] --> B[Preprocessing<br>Scaling · Encoding · NaN Handling]
    B --> C[Pipeline API<br>fit · transform · predict]
    C --> D[Streaming Decision Tree]
    C --> E[Streaming Random Forest]
    D --> F[Predictions]
    E --> F
    F --> G[Metrics<br>Accuracy · F1 · MSE]
    G --> H[Visualisation Module<br>Accuracy Curves · Chunk Tracking]
🔄 Streaming Workflow

sequenceDiagram
    participant U as Data Stream
    participant P as Preprocessing
    participant M as Model (Tree/Forest)
    participant T as Metrics
    participant V as Visualiser

    U->>P: Load next chunk
    P->>M: Transform + Update Model
    M->>T: Predict + Update Metrics
    T->>V: Update plots
    V->>U: Ready for next chunk
📊 Model Comparison (Tree vs Forest)

graph LR
    A[Decision Tree<br>Accuracy: 0.9067<br>Fit: 1.07s<br>Predict: 0.0006s]
    B[Random Forest<br>Accuracy: 0.7067<br>Fit: 3.14s<br>Predict: 0.0037s]

    A ---|Streaming| B
⚡ Vectorisation Speedup (Loop vs NumPy)

graph TD
    L[Python Loops<br>Mean: 0.0957s<br>Var: 0.3378s]
    V[Vectorised NumPy<br>Mean: 0.00044s<br>Var: 0.00469s]

    L -->|217× faster| V
🚀 Features
📥 Data I/O
Chunked CSV loader

Missing‑value handling

🧼 Preprocessing (Streaming‑Compatible)
StandardScaler, MinMaxScaler

One‑Hot Encoding

NaN‑safe transformations

🔍 Sorting & Searching
Stable + multi‑key sorting

Top‑k via argpartition

Quickselect

Binary search

🏆 Ranking
Average, dense, ordinal ranking

Percentiles

📊 Statistics (Incremental)
Running mean/variance (Welford)

Quantiles

NaN‑robust operations

📏 Metrics
Accuracy, Precision, Recall, F1

Confusion matrix

MSE, RMSE, MAD, MAPE

Streaming metric tracking
🌲 Models
Streaming Decision Tree

Streaming Random Forest

Deterministic splits + tie handling

🔗 Pipeline API
Transformer/Estimator chaining

Batch + streaming compatibility
📈 Visualisation Module
Chunk accuracy

Cumulative accuracy

Predictions vs ground truth

Training/prediction time
📁 Project Structure
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
⚙️ Installation
git clone <your-repo-link>
cd NumCompute
pip install -e .

Run demo:
python -m demo

🧪 Testing (unchanged)
pip install pytest
python -m pytest tests -q

Vectorised vs loop:
python benchmark/benchmark_vectorised_vs_loop.py

Tree vs forest:
python benchmark/benchmark_tree_vs_forest.py

👥 Author
Sanzana Mahrukh Hassan

---

If you want, I can also generate:

✨ A **logo** for NumCompute‑Stream  
✨ A **badge section** (coverage, license, build)  
✨ A **gallery section** with your visualisation plots  
✨ A **pip‑installable package layout**  

Just tell me what you want next.
