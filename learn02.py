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