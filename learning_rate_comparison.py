
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

# --------------------------------------------------
# 2. Generate Synthetic NoC Dataset
# --------------------------------------------------

samples = 1200

traffic = np.random.uniform(10, 100, samples)
buffer = np.random.uniform(4, 64, samples)
pe = np.random.uniform(16, 128, samples)
hop = np.random.uniform(1, 8, samples)
frequency = np.random.uniform(400, 1200, samples)
voltage = np.random.uniform(0.6, 1.2, samples)

X = np.column_stack((
    traffic,
    buffer,
    pe,
    hop,
    frequency,
    voltage
))

# Synthetic energy function (illustrative only)
noise = np.random.normal(0, 1.5, samples)

y = (
    0.25 * traffic
    + 0.15 * buffer
    + 0.20 * pe
    + 1.8 * hop
    + 0.004 * frequency
    + 12 * voltage
    + noise
)

y = y.reshape(-1, 1)

# --------------------------------------------------
# 3. Shuffle and Split Dataset
# --------------------------------------------------

indices = np.random.permutation(samples)

X = X[indices]
y = y[indices]

split_index = int(samples * 0.8)

X_train = X[:split_index]
X_val = X[split_index:]

y_train = y[:split_index]
y_val = y[split_index:]

# --------------------------------------------------
# 4. Standardization
# --------------------------------------------------

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std

y_mean = y_train.mean()
y_std = y_train.std()

y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std

# --------------------------------------------------
# 5. Create Neural Network
# --------------------------------------------------

def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(1)
    ])

    return model


# --------------------------------------------------
# 6. Prepare Identical Initial Weights
# --------------------------------------------------

base_model = create_model()
initial_weights = base_model.get_weights()

# --------------------------------------------------
# 7. Compare Learning Rates
# --------------------------------------------------

learning_rates = [0.1, 0.01, 0.001]

epochs = 100

histories = {}

for lr in learning_rates:

    print(f"\nTraining with Learning Rate: {lr}")

    model = create_model()

    # Use identical initial weights
    model.set_weights(initial_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr
        ),
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        verbose=0
    )

    histories[lr] = history.history

    print("First 5 Validation Loss values:")
    print(history.history["val_loss"][:5])

    print("Last 5 Validation Loss values:")
    print(history.history["val_loss"][-5:])
    # --------------------------------------------------
    # 8. Print Evaluation Results
    # --------------------------------------------------
    # --------------------------------------------------
    # --------------------------------------------------

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

    print(f"Train Loss: {train_loss:.6f}")
    print(f"Train MAE:  {train_mae:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation MAE:  {val_mae:.6f}")


# --------------------------------------------------
# 9. Plot Loss Curves
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for lr in learning_rates:

    plt.plot(
        histories[lr]["loss"],
        label=f"Train LR={lr}"
    )

    plt.plot(
        histories[lr]["val_loss"],
        linestyle="--",
        label=f"Validation LR={lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Learning Rate Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
