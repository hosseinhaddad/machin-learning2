
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Prepare the data
X = np.array(
    [0, 1, 2, 3, 4, 5],
    dtype=np.float32
)

y = np.array(
    [1, 3, 5, 7, 9, 11],
    dtype=np.float32
)

# 2. Build the model
model = keras.Sequential([
    keras.Input(shape=(1,)),
    layers.Dense(1, activation="linear")
])

# 3. Configure the learning process
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss="mse",
    metrics=["mae"]
)

# 4. Train the model
history = model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

# 5. Predict new values
X_new = np.array([6, 7], dtype=np.float32)

y_pred = model.predict(X_new, verbose=0)

# 6. Display results
print("Final Loss:", history.history["loss"][-1])
print("Final MAE:", history.history["mae"][-1])

print("\nPredictions:")
for x_value, predicted in zip(X_new, y_pred.flatten()):
    print(f"x = {x_value:.0f} | Predicted y = {predicted:.3f}")
