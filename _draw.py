


import matplotlib.pyplot as plt

def draw_loss(history):

    loss_values = history.history["loss"]

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(loss_values) + 1),
        loss_values,
        label="Training Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")

    plt.legend()
    plt.grid(True)
    plt.show()

