import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# =========================
# Configuration
# =========================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001


# =========================
# Paths
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.dirname(current_dir)

dataset_path = os.path.join(
    project_dir,
    "noc_energy_dataset.csv"
)

figure_path = os.path.join(
    current_dir,
    "baseline_loss.png"
)

results_path = os.path.join(
    current_dir,
    "baseline_results.txt"
)


# =========================
# Load Dataset
# =========================

df = pd.read_csv(dataset_path)

features = [
    "Traffic",
    "Buffer",
    "PE",
    "Hop",
    "Frequency",
    "Voltage"
]

target = "Energy"

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32)


# =========================
# Train / Validation / Test
# =========================

samples = len(X)

indices = np.random.permutation(samples)

train_end = int(samples * 0.80)
validation_end = int(samples * 0.90)

train_indices = indices[:train_end]
validation_indices = indices[train_end:validation_end]
test_indices = indices[validation_end:]

X_train = X[train_indices]
y_train = y[train_indices]

X_validation = X[validation_indices]
y_validation = y[validation_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# =========================
# Standardization
# =========================

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

X_train = (X_train - train_mean) / train_std
X_validation = (X_validation - train_mean) / train_std
X_test = (X_test - train_mean) / train_std


# =========================
# Build Baseline Model
# =========================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(1)
])


# =========================
# Compile Model
# =========================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)


# =========================
# Train Model
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# =========================
# Evaluate Model
# =========================

train_loss, train_mae = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

validation_loss, validation_mae = model.evaluate(
    X_validation,
    y_validation,
    verbose=0
)

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


# =========================
# Print Results
# =========================

print("\n==============================")
print("Baseline Results")
print("==============================")

print(f"Train Loss: {train_loss:.4f}")
print(f"Train MAE: {train_mae:.4f}")

print(f"Validation Loss: {validation_loss:.4f}")
print(f"Validation MAE: {validation_mae:.4f}")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")


# =========================
# Save Results to TXT
# =========================

with open(results_path, "w", encoding="utf-8") as file:

    file.write("==============================\n")
    file.write("Baseline Results\n")
    file.write("==============================\n\n")

    file.write(f"Epochs: {EPOCHS}\n")
    file.write(f"Batch Size: {BATCH_SIZE}\n")
    file.write(f"Learning Rate: {LEARNING_RATE}\n\n")

    file.write("Model Architecture:\n")
    file.write("Input(6)\n")
    file.write("Dense(128, ReLU)\n")
    file.write("Dense(128, ReLU)\n")
    file.write("Dense(64, ReLU)\n")
    file.write("Dense(1)\n\n")

    file.write("Final Evaluation Results:\n\n")

    file.write(f"Train Loss: {train_loss:.4f}\n")
    file.write(f"Train MAE: {train_mae:.4f}\n\n")

    file.write(
        f"Validation Loss: {validation_loss:.4f}\n"
    )

    file.write(
        f"Validation MAE: {validation_mae:.4f}\n\n"
    )

    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test MAE: {test_mae:.4f}\n")


# =========================
# Plot Loss Curve
# =========================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Baseline Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print("\nResults saved to:")
print(results_path)

print("\nLoss curve saved to:")
print(figure_path)