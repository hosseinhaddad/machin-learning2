
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt


# =========================
# Start Timer
# =========================

start_time = time.perf_counter()
execution_datetime = datetime.now()


# =========================
# Paths
# =========================

current_dir = os.path.dirname(os.path.abspath(__file__))

mae_figure_path = os.path.join(
    current_dir,
    "comparison_mae.png"
)

loss_figure_path = os.path.join(
    current_dir,
    "comparison_loss.png"
)

results_path = os.path.join(
    current_dir,
    "comparison_results.txt"
)


# =========================
# Experimental Results
# =========================

results = {
    "Baseline": {
        "Train Loss": 4.0697,
        "Train MAE": 1.5937,
        "Validation Loss": 7.4934,
        "Validation MAE": 2.0638,
        "Test Loss": 4.4193,
        "Test MAE": 1.7026
    },

    "Batch Normalization": {
        "Train Loss": 5.2716,
        "Train MAE": 1.8317,
        "Validation Loss": 9.4775,
        "Validation MAE": 2.3974,
        "Test Loss": 6.2683,
        "Test MAE": 1.9253
    }
}


# =========================
# Comparison Data
# =========================

models = list(results.keys())

mae_metrics = [
    "Train MAE",
    "Validation MAE",
    "Test MAE"
]

loss_metrics = [
    "Train Loss",
    "Validation Loss",
    "Test Loss"
]


# =========================
# MAE Comparison
# =========================

x = range(len(mae_metrics))
width = 0.35

baseline_mae = [
    results["Baseline"][metric]
    for metric in mae_metrics
]

bn_mae = [
    results["Batch Normalization"][metric]
    for metric in mae_metrics
]

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline_mae,
    width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    bn_mae,
    width,
    label="Batch Normalization"
)

plt.xticks(
    list(x),
    mae_metrics
)

plt.ylabel("MAE")
plt.title("MAE Comparison")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    mae_figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================
# Loss Comparison
# =========================

baseline_loss = [
    results["Baseline"][metric]
    for metric in loss_metrics
]

bn_loss = [
    results["Batch Normalization"][metric]
    for metric in loss_metrics
]

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline_loss,
    width,
    label="Baseline"
)

plt.bar(
    [i + width / 2 for i in x],
    bn_loss,
    width,
    label="Batch Normalization"
)

plt.xticks(
    list(x),
    loss_metrics
)

plt.ylabel("MSE Loss")
plt.title("Loss Comparison")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    loss_figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================
# Calculate Differences
# =========================

comparison_metrics = mae_metrics + loss_metrics

differences = {}

for metric in comparison_metrics:

    baseline_value = results["Baseline"][metric]

    bn_value = results["Batch Normalization"][metric]

    percentage = (
        (bn_value - baseline_value)
        / baseline_value
    ) * 100

    differences[metric] = percentage


# =========================
# Determine Best Model
# =========================

baseline_test_mae = results["Baseline"]["Test MAE"]

bn_test_mae = results["Batch Normalization"]["Test MAE"]

if baseline_test_mae < bn_test_mae:
    best_model = "Baseline"
else:
    best_model = "Batch Normalization"


# =========================
# Print Results
# =========================

print("\n==============================")
print("Model Comparison")
print("==============================")

for metric in comparison_metrics:

    print(f"\n{metric}")

    print(
        f"Baseline: "
        f"{results['Baseline'][metric]:.4f}"
    )

    print(
        f"Batch Normalization: "
        f"{results['Batch Normalization'][metric]:.4f}"
    )

    print(
        f"Difference: "
        f"{differences[metric]:+.2f}%"
    )

print("\nBest Model Based on Test MAE:")
print(best_model)


# =========================
# Save Report
# =========================

with open(
    results_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("==============================\n")
    file.write("Model Comparison Report\n")
    file.write("==============================\n\n")

    file.write(
        f"Execution Date: "
        f"{execution_datetime.strftime('%Y-%m-%d')}\n"
    )

    file.write(
        f"Execution Time: "
        f"{execution_datetime.strftime('%H:%M:%S')}\n\n"
    )

    file.write("Experimental Results:\n\n")

    for metric in comparison_metrics:

        file.write(f"{metric}\n")

        file.write(
            f"Baseline: "
            f"{results['Baseline'][metric]:.4f}\n"
        )

        file.write(
            f"Batch Normalization: "
            f"{results['Batch Normalization'][metric]:.4f}\n"
        )

        file.write(
            f"Difference: "
            f"{differences[metric]:+.2f}%\n\n"
        )

    file.write("==============================\n")
    file.write("Conclusion\n")
    file.write("==============================\n\n")

    file.write(
        f"Best Model Based on Test MAE: "
        f"{best_model}\n"
    )

    file.write(
        "Lower Loss and MAE indicate better performance.\n"
    )


# =========================
# Execution Duration
# =========================

execution_duration = time.perf_counter() - start_time

execution_minutes = execution_duration / 60


# =========================
# Append Execution Information
# =========================

with open(
    results_path,
    "a",
    encoding="utf-8"
) as file:

    file.write("\n==============================\n")
    file.write("Execution Information\n")
    file.write("==============================\n")

    file.write(
        f"Execution Duration: "
        f"{execution_duration:.4f} seconds\n"
    )

    file.write(
        f"Execution Duration: "
        f"{execution_minutes:.4f} minutes\n"
    )


# =========================
# Final Messages
# =========================

print("\nExecution Information")
print("------------------------------")

print(
    f"Duration: "
    f"{execution_duration:.4f} seconds"
)

print("\nFiles Saved:")
print(results_path)
print(mae_figure_path)
print(loss_figure_path)
