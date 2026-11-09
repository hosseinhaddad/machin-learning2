
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Dataset configuration
samples = 1000

# Generate input features
Traffic = np.random.uniform(10, 100, samples)
Buffer = np.random.uniform(4, 64, samples)
PE = np.random.uniform(10, 100, samples)
Hop = np.random.uniform(1, 8, samples)
Frequency = np.random.uniform(200, 1000, samples)
Voltage = np.random.uniform(0.6, 1.2, samples)

# Generate target values
Energy = (
    0.15 * Traffic +
    0.08 * Buffer +
    0.12 * PE +
    0.9 * Hop +
    0.01 * Frequency +
    5.0 * Voltage
)

# Add random noise
Energy += np.random.normal(0, 2, samples)

# Create DataFrame
data = pd.DataFrame({
    "Traffic": Traffic,
    "Buffer": Buffer,
    "PE": PE,
    "Hop": Hop,
    "Frequency": Frequency,
    "Voltage": Voltage,
    "Energy": Energy
})

# Save dataset
data.to_csv("noc_energy_dataset.csv", index=False)

print("Dataset created successfully!")
print("\nDataset Information:")
print(data.describe())

print("\nFirst 5 Samples:")
print(data.head())
