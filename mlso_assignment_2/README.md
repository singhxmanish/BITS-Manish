# MLSO Assignment 2 — Spark-Based XGBoost Parallel Training

This repository contains an experimental study of **serial vs. Spark-parallel training** of XGBoost for binary classification using the Credit Card Fraud Detection dataset.

The primary focus of the assignment is **machine learning systems optimization**: measuring how Spark-based data parallelism improves training time relative to sequential execution.

---

## 📁 Project Structure

```
mlso_assignment_2/
│
├── data/                     # Dataset storage
│   └── creditcard.csv
│
├── results/                  # Output figures
│   ├── speedup.png
│   └── speedup.pdf
│
├── logs/                     # Execution logs
│   └── run.log
│
├── src/
│   ├── driver.py             # Main experiment runner
│   ├── data_loader.py       # Dataset loading & preprocessing
│   ├── train_serial.py      # Serial training
│   ├── train_parallel.py    # Spark-parallel training
│   ├── inference.py         # Evaluation logic
│   └── metrics.py           # Accuracy / Precision / Recall / F1
│
└── README.md
```

---

## ⚙️ Environment Setup

### Python Version

Recommended:

```
Python 3.10 or 3.11
```

---

### Install Dependencies

Create and activate a virtual environment:

```bash
python3.10 -m venv spark_env
source spark_env/bin/activate
```

Install required packages:

```bash
pip install pyspark==3.5.1 xgboost kagglehub matplotlib
```

---

### Java Requirement

Spark requires Java 11 or newer:

```bash
java -version
```

---

## ▶️ How to Run

From the project root:

```bash
cd mlso_assignment_2
python src/driver.py
```

---

## 📊 What the Driver Script Does

`driver.py` orchestrates the full pipeline:

1. Creates Spark session
2. Loads dataset (local-first, Kaggle fallback)
3. Runs serial training
4. Runs Spark-parallel training
5. Computes training-time metrics
6. Saves IEEE-ready plots
7. Writes logs to disk
8. Stops Spark cleanly

---


# ⚡ Performance Metrics

The main objective of this assignment is to **quantify the runtime improvement achieved through Spark-based data parallelism**.

While predictive metrics are computed for correctness verification, **training-time speedup is the primary systems-level metric analyzed and reported**.

---

## 📐 Metrics Computed

### 🔹 1. Serial Training Time (`T_serial`)

Time required to train the XGBoost model when:

* The dataset is repartitioned into a small number of partitions
* Spark executes tasks sequentially
* Represents baseline execution

Measured in:

```
seconds
```

---

### 🔹 2. Parallel Training Time (`T_parallel`)

Time required to train the same model when:

* Data is repartitioned into 8 partitions
* Spark schedules tasks concurrently across CPU cores
* Demonstrates data-parallel execution

Measured in:

```
seconds
```

---

### 🔹 3. Speedup Factor (`S`)

Speedup measures the relative benefit of parallel execution and is defined as:

```
S = T_serial / T_parallel
```

Where:

* `T_serial` = serial training time
* `T_parallel` = parallel training time

Interpretation:

* `S > 1` → performance improvement
* `S = 1` → no benefit
* `S < 1` → parallel overhead dominates

---

### 🔹 4. Visual Speedup Analysis

Two plots are generated automatically:

* **`speedup.png`** — bar chart comparing serial and parallel runtimes
* **`speedup.pdf`** — IEEE-ready vector plot suitable for academic reports

Saved in:

```
results/
```

---

## 📝 Author
MANISH KUMAR SINGH

BADINENI SAIDATTA

SAHIL VERMA

SHUBHANJOY BISWAS

VAKACHARLA SA CH RAO


---
