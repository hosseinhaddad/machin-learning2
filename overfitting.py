
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ---------------------------------
# 1. Generate Dataset
# ---------------------------------

np.random.seed(42)
tf.random.set_seed(42)

X = np.linspace(-3, 3, 200)

y = X ** 2 + np.random.normal(0, 0.5, 200)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


# ---------------------------------
# 2. Split Dataset
# ---------------------------------

# First split: 80% Train + Validation, 20% Test

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Second split: 80% Train, 20% Validation
# Relative to the remaining 80%

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2,
    random_state=42
)


# ---------------------------------
# 3. Build Neural Network
# ---------------------------------

model = keras.Sequential([
    layers.Input(shape=(1,)),

    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),

    layers.Dense(1)
])


# ---------------------------------
# 4. Compile Model
# ---------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.01
    ),
    loss="mse",
    metrics=["mae"]
)


# ---------------------------------
# 5. Train Model
# ---------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=16,
    verbose=0
)


# ---------------------------------
# 6. Evaluate Model
# ---------------------------------

train_loss, train_mae = model.evaluate(
    X_train, y_train, verbose=0
)

val_loss, val_mae = model.evaluate(
    X_val, y_val, verbose=0
)

test_loss, test_mae = model.evaluate(
    X_test, y_test, verbose=0
)


# ---------------------------------
# 7. Print Results
# ---------------------------------

print("\nDataset Sizes")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nEvaluation Results")

print("Train Loss:", train_loss)
print("Train MAE:", train_mae)

print("Validation Loss:", val_loss)
print("Validation MAE:", val_mae)

print("Test Loss:", test_loss)
print("Test MAE:", test_mae)


# ---------------------------------
# 8. Plot Loss
# ---------------------------------

plt.figure(figsize=(9, 5))

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
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
