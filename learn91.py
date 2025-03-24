# Import required libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a Larger Dataset
np.random.seed(42)  # For reproducibility
n_samples = 50  # Increase to 50 samples for better class distribution
ph_values = np.random.uniform(5.0, 8.0, n_samples)       # Soil pH (5.0–8.0)
moisture_values = np.random.uniform(10, 80, n_samples)   # Moisture % (10–80)
X = np.column_stack((ph_values, moisture_values))        # Feature matrix

# Labels: 1 = suitable for tomatoes (pH 6.0–7.0, moisture 30–60), 0 = unsuitable
y = np.array([1 if (6.0 <= ph <= 7.0 and 30 <= moist <= 60) else 0 
              for ph, moist in X])

print("Feature Matrix (X) Shape:", X.shape)
print("Labels (y):", y)
print("Number of Suitable Plots (1):", np.sum(y))
print("Number of Unsuitable Plots (0):", np.sum(y == 0))

# Step 2: Normalize Data
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Step 3: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

print("\nTraining Features Shape:", X_train.shape)
print("Training Labels:", y_train)
print("Unique Classes in y_train:", np.unique(y_train))

# Step 4: Check for Single-Class Issue and Train Model
if len(np.unique(y_train)) < 2:
    print("\nWarning: y_train contains only one class. Logistic Regression requires at least two classes.")
    print("Try increasing the dataset size or adjusting the suitability conditions.")
else:
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 5: Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", accuracy)

    # Bonus: Predict suitability for a new plot
    new_plot = np.array([[6.5, 45]])  # pH 6.5, 45% moisture
    new_plot_normalized = (new_plot - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    prediction = model.predict(new_plot_normalized)
    print("New Plot Suitability (1 = suitable, 0 = unsuitable):", prediction[0])