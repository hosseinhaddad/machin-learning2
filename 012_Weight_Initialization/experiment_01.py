import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SEED = 42
SAMPLES = 1000
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Start Time
# ============================================================

program_start_time = time.perf_counter()
start_datetime = datetime.now()


# ============================================================
# Create Dataset
# ============================================================

X = np.random.uniform(
    -1,
    1,
    (SAMPLES, 3)
)

y = (
    3 * X[:, 0]
    + 2 * X[:, 1]
    - 4 * X[:, 2]
    + 1
)

y = y.reshape(-1, 1)


# ============================================================
# Model Creation Function
# ============================================================

def create_model(initializer):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(3,)),

        tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_initializer=initializer
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_initializer=initializer
        ),

        tf.keras.layers.Dense(
            1,
            kernel_initializer="zeros"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# Experiment 1
# Zero Initialization
# ============================================================

print("=" * 60)
print("Experiment 1: Zero Initialization")
print("=" * 60)

zero_start_time = time.perf_counter()

zero_model = create_model("zeros")

zero_history = zero_model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0
)

zero_end_time = time.perf_counter()

zero_duration = zero_end_time - zero_start_time


# ============================================================
# Experiment 2
# Random Initialization
# ============================================================

print()
print("=" * 60)
print("Experiment 2: Random Initialization")
print("=" * 60)

random_start_time = time.perf_counter()

random_model = create_model(
    tf.keras.initializers.RandomNormal(
        mean=0.0,
        stddev=0.05
    )
)

random_history = random_model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=0
)

random_end_time = time.perf_counter()

random_duration = random_end_time - random_start_time


# ============================================================
# Extract Results
# ============================================================

zero_train_loss = zero_history.history["loss"][-1]
zero_val_loss = zero_history.history["val_loss"][-1]

zero_train_mae = zero_history.history["mae"][-1]
zero_val_mae = zero_history.history["val_mae"][-1]

random_train_loss = random_history.history["loss"][-1]
random_val_loss = random_history.history["val_loss"][-1]

random_train_mae = random_history.history["mae"][-1]
random_val_mae = random_history.history["val_mae"][-1]


# ============================================================
# End Time
# ============================================================

program_end_time = time.perf_counter()
end_datetime = datetime.now()

total_duration = program_end_time - program_start_time


# ============================================================
# Console Output
# ============================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print()
print("Zero Initialization")
print(f"Training Loss:   {zero_train_loss:.6f}")
print(f"Validation Loss: {zero_val_loss:.6f}")
print(f"Training MAE:    {zero_train_mae:.6f}")
print(f"Validation MAE:  {zero_val_mae:.6f}")
print(f"Execution Time:  {zero_duration:.4f} seconds")

print()
print("Random Initialization")
print(f"Training Loss:   {random_train_loss:.6f}")
print(f"Validation Loss: {random_val_loss:.6f}")
print(f"Training MAE:    {random_train_mae:.6f}")
print(f"Validation MAE:  {random_val_mae:.6f}")
print(f"Execution Time:  {random_duration:.4f} seconds")

print()
print("=" * 60)
print("PROGRAM INFORMATION")
print("=" * 60)

print(f"Start Time: {start_datetime}")
print(f"End Time:   {end_datetime}")
print(f"Total Execution Time: {total_duration:.4f} seconds")


# ============================================================
# Save Loss Curve
# ============================================================

plot_path = os.path.join(
    RESULTS_DIR,
    "experiment_01_comparison.png"
)

plt.figure(figsize=(10, 6))

plt.plot(
    zero_history.history["loss"],
    label="Zero - Training"
)

plt.plot(
    zero_history.history["val_loss"],
    label="Zero - Validation"
)

plt.plot(
    random_history.history["loss"],
    label="Random - Training"
)

plt.plot(
    random_history.history["val_loss"],
    label="Random - Validation"
)

plt.title("Weight Initialization Comparison")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()


# ============================================================
# Generate Report
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "experiment_01_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write("=" * 60 + "\n")
    report.write("DEEP LEARNING - WEIGHT INITIALIZATION\n")
    report.write("Experiment 01: Zero vs Random Initialization\n")
    report.write("=" * 60 + "\n\n")

    report.write("PROGRAM INFORMATION\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Start Time: {start_datetime}\n"
    )

    report.write(
        f"End Time: {end_datetime}\n"
    )

    report.write(
        f"Total Execution Time: {total_duration:.4f} seconds\n\n"
    )

    report.write("EXPERIMENT CONFIGURATION\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Random Seed: {SEED}\n"
    )

    report.write(
        f"Samples: {SAMPLES}\n"
    )

    report.write(
        f"Epochs: {EPOCHS}\n"
    )

    report.write(
        f"Batch Size: {BATCH_SIZE}\n"
    )

    report.write(
        f"Learning Rate: {LEARNING_RATE}\n\n"
    )

    report.write("ZERO INITIALIZATION\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Training Loss: {zero_train_loss:.6f}\n"
    )

    report.write(
        f"Validation Loss: {zero_val_loss:.6f}\n"
    )

    report.write(
        f"Training MAE: {zero_train_mae:.6f}\n"
    )

    report.write(
        f"Validation MAE: {zero_val_mae:.6f}\n"
    )

    report.write(
        f"Execution Time: {zero_duration:.4f} seconds\n\n"
    )

    report.write("RANDOM INITIALIZATION\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Training Loss: {random_train_loss:.6f}\n"
    )

    report.write(
        f"Validation Loss: {random_val_loss:.6f}\n"
    )

    report.write(
        f"Training MAE: {random_train_mae:.6f}\n"
    )

    report.write(
        f"Validation MAE: {random_val_mae:.6f}\n"
    )

    report.write(
        f"Execution Time: {random_duration:.4f} seconds\n\n"
    )

    report.write("RESULT FILES\n")
    report.write("-" * 60 + "\n")

    report.write(
        f"Loss Curve: {plot_path}\n"
    )

    report.write(
        f"Report: {report_path}\n"
    )

    report.write("\n")
    report.write("=" * 60 + "\n")
    report.write("END OF REPORT\n")
    report.write("=" * 60 + "\n")


# ============================================================
# Final Message
# ============================================================

print()
print("=" * 60)
print("Experiment completed successfully.")
print(f"Report saved to: {report_path}")
print(f"Plot saved to:   {plot_path}")
print("=" * 60)