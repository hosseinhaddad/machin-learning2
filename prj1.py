import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from _draw import draw_loss
x=int(input("Enter Epoch?"))
if x<=0 : 
    EPOCH=100; 
else: 
    EPOCH=x

EPOCH=100;

tf.keras.utils.set_random_seed(42)
# 1. Prepare the data
X = np.array(
    [0, 1, 2, 3, 4, 5,6,7,8,9,10],
    dtype=np.float32
)

y = np.array(
    [1, 3, 5, 7, 9, 11,13,15,17,19,21],
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
    epochs=EPOCH,
    verbose=0
)

# 5. Predict new values
X_new = np.array([13, 17], dtype=np.float32)

y_pred = model.predict(X_new, verbose=0)

# 6. Display results

print("-------------------------------------------------")
print("Final Loss:", history.history["loss"][-1])
print("Final MAE:", history.history["mae"][-1])

print("Epoch:", EPOCH)
print("X:", X)
print("Y:", y)

print("\nPredictions:")
for x_value, predicted in zip(X_new, y_pred.flatten()):
    print(f"x = {x_value:.0f} | Predicted y = {predicted:.3f}")

print("-------------------------------------------------")
draw_loss(history)