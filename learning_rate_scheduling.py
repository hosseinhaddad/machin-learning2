
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate Dataset
samples = 1000

X = np.random.uniform(0, 100, samples)
noise = np.random.normal(0, 5, samples)

y = 2 * X + 10 + noise

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Split Dataset
split_index = int(samples * 0.8)

X_train = X[:split_index]
X_val = X[split_index:]

y_train = y[:split_index]
y_val = y[split_index:]


# Build Model
def build_model():

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    return model


# Training Configuration
epochs = 100
batch_size = 32

# Model 1: Fixed Learning Rate
model_fixed = build_model()

model_fixed.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)

history_fixed = model_fixed.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=0
)


# Model 2: Exponential Decay
initial_learning_rate = 0.001

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=500,
    decay_rate=0.9,
    staircase=False
)

model_decay = build_model()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule
)

model_decay.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)

history_decay = model_decay.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=0
)


# Calculate Learning Rate Values
steps_per_epoch = int(np.ceil(len(X_train) / batch_size))

learning_rates = [
    float(lr_schedule(epoch * steps_per_epoch).numpy())
    for epoch in range(epochs)
]


# Print Results
print("\nFixed Learning Rate")
print("Final Training Loss:", history_fixed.history["loss"][-1])
print("Final Validation Loss:", history_fixed.history["val_loss"][-1])

print("\nExponential Decay")
print("Final Training Loss:", history_decay.history["loss"][-1])
print("Final Validation Loss:", history_decay.history["val_loss"][-1])

print("\nLearning Rate Values")
print("First Epoch:", learning_rates[0])
print("Middle Epoch:", learning_rates[epochs // 2])
print("Last Epoch:", learning_rates[-1])


# Plot Loss Comparison
plt.figure(figsize=(10, 5))

plt.plot(
    history_fixed.history["val_loss"],
    label="Fixed LR"
)

plt.plot(
    history_decay.history["val_loss"],
    label="Exponential Decay"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Learning Rate Scheduling Comparison")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot Learning Rate Schedule
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, epochs + 1),
    learning_rates,
    label="Learning Rate"
)

plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Exponential Learning Rate Decay")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
