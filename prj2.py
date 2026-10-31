import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from _draw import draw_loss

EPOCH=100;
x=int(input("Enter Epoch?"))
if x<=0 : 
    EPOCH=100; 
else: 
    EPOCH=x



X = np.arange(0, 101, dtype=np.float32).reshape(-1, 1)

y = 2 * X + 1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


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