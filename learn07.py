import numpy as np


# Features: [pH, moisture %]
np.random.seed(42)
ph_values = np.random.uniform(5.0, 8.0, 10)
moisture_values = np.random.uniform(10, 80, 10)
X = np.column_stack((ph_values, moisture_values))  # Feature matrix

# Labels: 1 = suitable, 0 = unsuitable (based on tomato conditions)
y = np.array([1 if (6.0 <= ph <= 7.0 and 30 <= moist <= 60) else 0 
              for ph, moist in X])

print("Feature Matrix (X):")
print(X)
print("\nLabels (y):")
print(y)

X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("Normalized Features:")
print(X_normalized)

