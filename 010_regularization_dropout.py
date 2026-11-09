import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
data = pd.read_csv("noc_energy_dataset.csv")

# Separate features and target
X = data[
    [
        "Traffic",
        "Buffer",
        "PE",
        "Hop",
        "Frequency",
        "Voltage"
    ]
].values

y = data["Energy"].values

# Split dataset
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Build baseline model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# Display model architecture
model.summary()

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Evaluate model
train_loss, train_mae = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

val_loss, val_mae = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# Print results
print("\n===== Baseline Model Results =====")

print(f"Train Loss: {train_loss:.4f}")
print(f"Train MAE:  {train_mae:.4f}")

print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation MAE:  {val_mae:.4f}")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE:  {test_mae:.4f}")

# Plot training and validation loss
loss = history.history["loss"]
val_loss_history = history.history["val_loss"]

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(loss) + 1),
    loss,
    label="Training Loss"
)

plt.plot(
    range(1, len(val_loss_history) + 1),
    val_loss_history,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Baseline Model - Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.show()