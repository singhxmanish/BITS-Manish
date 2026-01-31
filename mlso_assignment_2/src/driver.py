"""
Main experiment driver.

Location:
    mlso_assignment_2/src/driver.py

Responsibilities:
- Force same Python for Spark driver/workers
- Create Spark session
- Load data
- Train serial and parallel XGBoost models
- Evaluate
- Save plots into mlso_assignment_2/results/
- Save logs into mlso_assignment_2/logs/
"""

import os
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT RESOLUTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "run.log"

# ============================================================
# TEE STDOUT / STDERR TO FILE + CONSOLE
# ============================================================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


log_fh = open(LOG_FILE, "w")

sys.stdout = Tee(sys.__stdout__, log_fh)
sys.stderr = Tee(sys.__stderr__, log_fh)

print(">>> Logging started")
print(">>> Log file:", LOG_FILE)

# ============================================================
# FORCE SAME PYTHON FOR SPARK DRIVER & WORKERS
# ============================================================

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print(">>> Project root:", PROJECT_ROOT)
print(">>> Python executable:", sys.executable)
print(">>> Python version:", sys.version)
print(">>> PYSPARK_PYTHON =", os.environ.get("PYSPARK_PYTHON"))
print(">>> PYSPARK_DRIVER_PYTHON =", os.environ.get("PYSPARK_DRIVER_PYTHON"))

# ============================================================
# CREATE SPARK SESSION
# ============================================================

print(">>> Creating SparkSession...")

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[8]")
    .appName("XGBoost-Spark-Local")
    .config("spark.eventLog.enabled", "true")
    .config("spark.eventLog.dir", str(LOGS_DIR))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print(">>> Spark created.")
print(">>> Spark master:", spark.sparkContext.master)
print(">>> Default parallelism:", spark.sparkContext.defaultParallelism)
print(">>> Spark UI:", spark.sparkContext.uiWebUrl)

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

print(">>> Importing project modules...")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_data
from train_serial import train_serial
from train_parallel import train_parallel
from inference import evaluate_model

print(">>> Imports completed.")

# ============================================================
# LOAD DATA
# ============================================================

print(">>> Loading dataset...")

train_df, test_df = load_data(spark)

print(">>> Dataset loaded.")

spark.catalog.clearCache()

# ============================================================
# TRAIN MODELS
# ============================================================

print(">>> Starting SERIAL training...")

serial_model, T_serial = train_serial(train_df)

print(f">>> Serial training done in {T_serial:.2f} s")

print(">>> Starting PARALLEL training...")

parallel_model, T_parallel = train_parallel(train_df)

print(f">>> Parallel training done in {T_parallel:.2f} s")

speedup = T_serial / T_parallel

# ============================================================
# EVALUATION
# ============================================================

print(">>> Evaluating parallel model...")

acc, prec, rec, f1, auc = evaluate_model(
    parallel_model,
    test_df
)

print(">>> Evaluation complete.")

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n===== FINAL RESULTS =====")
print(f"T_serial   : {T_serial:.2f} s")
print(f"T_parallel : {T_parallel:.2f} s")
print(f"Speedup    : {speedup:.2f} x")

print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"AUC-ROC    : {auc:.4f}")

# ============================================================
# SAVE PLOTS 
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(">>> Saving plots to:", RESULTS_DIR)

# ---------- Normal bar plot ----------
plt.figure(figsize=(6, 4))

plt.bar(["Serial", "Spark Parallel"], [T_serial, T_parallel])

plt.ylabel("Training Time (seconds)", fontsize=11)
plt.xlabel("Execution Mode", fontsize=11)
plt.title("Spark XGBoost Training Time", fontsize=12)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "speedup.png", dpi=300)
plt.close()

# ---------- IEEE plot ----------
plt.figure(figsize=(3.5, 2.5))

plt.plot(["Serial", "Parallel"], [T_serial, T_parallel], marker="o")

plt.ylabel("Training Time (s)", fontsize=9)
plt.xlabel("Mode", fontsize=9)
plt.title("Speedup from Spark Data Parallelism", fontsize=9)

plt.grid(True, linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "speedup.pdf",
    format="pdf",
    dpi=600
)
plt.close()

print(">>> Plots saved successfully.")

# ============================================================
# STOP SPARK
# ============================================================

print(">>> Stopping Spark...")

spark.stop()

print(">>> Done.")

log_fh.close()
