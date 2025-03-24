import numpy as np

# 1D array (vector)
soil_ph = np.array([6.5, 6.8, 7.0, 5.9])  # Soil pH levels for 4 garden plots
print("Soil pH:", soil_ph)

# 2D array (matrix)
soil_data = np.array([
    [6.5, 30],  # [pH, moisture %]
    [6.8, 45],
    [7.0, 25],
    [5.9, 60]
])
print("\nSoil Data (pH, Moisture):")
print(soil_data)

# Add 0.5 to all pH values (simulating a treatment)
adjusted_ph = soil_ph + 0.5
print("Adjusted pH:", adjusted_ph)

# Calculate mean soil moisture from the 2D array
mean_moisture = np.mean(soil_data[:, 1])  # Column 1 (moisture)
print("Mean Moisture:", mean_moisture)
print("Mean Moisture:", mean_moisture)