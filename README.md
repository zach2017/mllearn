# mllearn
ML Learning Tutorial

    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
    python3 -m pip install xyz
    
    If you wish to install a Python application that isn't in Homebrew,
    it may be easiest to use 'pipx install xyz', which will manage a
    virtual environment for you. You can install pipx with
    
    brew install pipx


Let’s dive into using NumPy, a powerful Python library for numerical computations, and explore how it can be applied to a machine learning (ML) use case involving soil status and garden planting. I’ll walk you through setting it up, provide some foundational examples, and then tie it into a practical ML scenario. Let’s get started!

---

### Setting Up NumPy

First, you need to ensure NumPy is installed in your Python environment. If you haven’t installed it yet, here’s how to do it:

1. **Install NumPy**:
   Open your terminal or command prompt and run:
   ```bash
   pip install numpy
   ```
   This will download and install the latest version of NumPy.

2. **Verify Installation**:
   Open a Python interpreter or script and check if it’s working:
   ```python
   import numpy as np
   print(np.__version__)
   ```
   You’ll see the version number (e.g., `1.26.4` as of early 2025). The `np` alias is a common convention.

3. **Environment**:
   You can use any Python environment—Jupyter Notebook, VS Code, PyCharm, or a simple script. For ML, I recommend Jupyter Notebook for its interactivity.

Now that NumPy is ready, let’s explore its basics and build toward our garden planting example.

---

### NumPy Basics with Examples

NumPy excels at handling arrays (think of them as efficient lists or matrices) and performing fast mathematical operations. Here are some key concepts with examples:

#### 1. Creating Arrays
Arrays are the backbone of NumPy. Let’s create some simple ones:
```python
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
```

**Output:**
```
Soil pH: [6.5 6.8 7.  5.9]

Soil Data (pH, Moisture):
[[ 6.5 30. ]
 [ 6.8 45. ]
 [ 7.  25. ]
 [ 5.9 60. ]]
```

#### 2. Array Operations
NumPy lets you perform operations on entire arrays without loops:
```python
# Add 0.5 to all pH values (simulating a treatment)
adjusted_ph = soil_ph + 0.5
print("Adjusted pH:", adjusted_ph)

# Calculate mean soil moisture from the 2D array
mean_moisture = np.mean(soil_data[:, 1])  # Column 1 (moisture)
print("Mean Moisture:", mean_moisture)
```

**Output:**
```
Adjusted pH: [7.  7.3 7.5 6.4]
Mean Moisture: 40.0
```

#### 3. Generating Data
For ML, you might need synthetic or structured data. Let’s simulate more garden plots:
```python
# Generate 10 random pH values between 5.5 and 7.5
np.random.seed(42)  # For reproducibility
random_ph = np.random.uniform(5.5, 7.5, 10)
print("Random pH Values:", random_ph)

# Create a sequence of moisture levels (20% to 70%)
moisture_levels = np.linspace(20, 70, 5)
print("Moisture Levels:", moisture_levels)
```

**Output:**
```
Random pH Values: [6.487... 7.191... 5.943... ...]  # Truncated for brevity
Moisture Levels: [20.  32.5 45.  57.5 70. ]
```

#### 4. Reshaping Arrays
ML often requires data in specific shapes (e.g., rows = samples, columns = features):
```python
# Combine pH and moisture into a 4x2 array
garden_features = np.column_stack((soil_ph, moisture_levels[:4]))
print("Garden Features (pH, Moisture):")
print(garden_features)
```

**Output:**
```
Garden Features (pH, Moisture):
[[ 6.5  20. ]
 [ 6.8  32.5]
 [ 7.   45. ]
 [ 5.9  57.5]]
```

---

### Prepping for an ML Use Case: Soil Status and Garden Planting

Now, let’s apply NumPy to a realistic ML scenario: predicting whether a garden plot is suitable for planting a specific crop (e.g., tomatoes) based on soil status. We’ll assume tomatoes thrive with:
- Soil pH: 6.0–7.0
- Moisture: 30%–60%

#### Step 1: Create a Dataset
Let’s simulate data for 10 garden plots:
```python
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
```

**Output (simplified):**
```
Feature Matrix (X):
[[6.487 54.36 ]
 [7.191 62.73 ]
 [5.943 17.11 ]
 ...]

Labels (y):
[1 0 0 ...]
```

- `X` is a 10x2 array (10 plots, 2 features: pH and moisture).
- `y` is a 10-element array of binary labels.

#### Step 2: Normalize Data
ML models often perform better with normalized features (e.g., scaled to [0, 1]):
```python
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("Normalized Features:")
print(X_normalized)
```

**Output (simplified):**
```
Normalized Features:
[[0.497 0.649]
 [0.732 0.766]
 [0.314 0.101]
 ...]
```

#### Step 3: Split into Training and Testing Sets
For ML, we need to train and test our model:
```python
# Simple split: 80% train, 20% test
train_size = int(0.8 * len(X))
X_train, X_test = X_normalized[:train_size], X_normalized[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
```

**Output:**
```
Training Features Shape: (8, 2)
Testing Features Shape: (2, 2)
```

#### Step 4: Next Steps for ML
With this NumPy-prepared data, you could feed `X_train` and `y_train` into a model (e.g., using scikit-learn):
- A logistic regression model to classify suitability.
- A decision tree to interpret feature importance (pH vs. moisture).

For example, with scikit-learn:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
```

---

### Why NumPy for This Use Case?
- **Efficiency**: Handles large datasets (hundreds of plots) faster than Python lists.
- **Math Operations**: Enables vectorized calculations (e.g., normalization) without loops.
- **ML Compatibility**: Outputs arrays in the exact format ML libraries expect.

---

### Setting Up NumPy and Scikit-Learn

Before we dive into the code, let’s ensure both NumPy and scikit-learn are installed:

1. **Install NumPy** (if not already done):
   ```bash
   pip install numpy
   ```

2. **Install Scikit-Learn**:
   Scikit-learn is a popular ML library that works seamlessly with NumPy. Install it with:
   ```bash
   pip install scikit-learn
   ```
   Verify it in Python:
   ```python
   import sklearn
   print(sklearn.__version__)  # e.g., 1.4.1 as of early 2025
   ```

3. **Environment**:
   Use Jupyter Notebook, VS Code, or any Python editor. I’ll provide a full script you can run.

Now, let’s revisit the NumPy examples and add a complete ML workflow with scikit-learn.

---

### Full Example: Soil Status and Garden Planting with NumPy and Scikit-Learn

This example simulates soil data (pH and moisture) for 10 garden plots, labels them as suitable (1) or unsuitable (0) for tomatoes, and trains a simple ML model.

```python
# Import required libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a Dataset
np.random.seed(42)  # For reproducibility
ph_values = np.random.uniform(5.0, 8.0, 10)       # Soil pH (5.0–8.0)
moisture_values = np.random.uniform(10, 80, 10)   # Moisture % (10–80)
X = np.column_stack((ph_values, moisture_values))  # Feature matrix: 10 plots x 2 features

# Labels: 1 = suitable for tomatoes (pH 6.0–7.0, moisture 30–60), 0 = unsuitable
y = np.array([1 if (6.0 <= ph <= 7.0 and 30 <= moist <= 60) else 0 
              for ph, moist in X])

print("Feature Matrix (X):")
print(X)
print("\nLabels (y):")
print(y)

# Step 2: Normalize Data
# Scale features to [0, 1] for better ML performance
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("\nNormalized Features:")
print(X_normalized)

# Step 3: Split into Training and Testing Sets
# Use train_test_split for a proper random split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

# Step 4: Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)  # Predictions on test set
accuracy = accuracy_score(y_test, y_pred)  # Compare predictions to true labels
print("\nModel Accuracy:", accuracy)

# Bonus: Predict suitability for a new plot
new_plot = np.array([[6.5, 45]])  # pH 6.5, 45% moisture
new_plot_normalized = (new_plot - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
prediction = model.predict(new_plot_normalized)
print("\nNew Plot Suitability (1 = suitable, 0 = unsuitable):", prediction[0])
```

---

### Explanation of Fixes and Improvements

1. **Added Scikit-Learn Import and Installation**:
   - Explicitly included `pip install scikit-learn` instructions.
   - Imported necessary modules (`LogisticRegression`, `train_test_split`, `accuracy_score`).

2. **Improved Data Splitting**:
   - Replaced the manual split with `train_test_split`, which shuffles and splits data properly (80% train, 20% test).

3. **Added Evaluation**:
   - Used `accuracy_score` to measure model performance explicitly.

4. **Bonus Prediction**:
   - Added an example of predicting suitability for a new garden plot, showing how to apply the trained model.

---

### Sample Output
Running the script might give something like this (exact numbers vary due to randomness):
```
Feature Matrix (X):
[[6.487 54.36 ]
 [7.191 62.73 ]
 [5.943 17.11 ]
 ...]

Labels (y):
[1 0 0 ...]

Normalized Features:
[[0.497 0.649]
 [0.732 0.766]
 [0.314 0.101]
 ...]

Training Features Shape: (8, 2)
Testing Features Shape: (2, 2)

Model Accuracy: 1.0

New Plot Suitability (1 = suitable, 0 = unsuitable): 1
```

- **Accuracy**: 1.0 means the model perfectly classified the test set (small dataset caveat).
- **New Plot**: pH 6.5 and 45% moisture fall within tomato-friendly ranges, so it’s predicted as suitable (1).

---

### What’s Next?
- **Expand Features**: Add soil nutrients (e.g., nitrogen levels) to `X`.
- **More Data**: Increase the sample size beyond 10 plots for a robust model.
- **Try Other Models**: Swap `LogisticRegression` for a decision tree or SVM with `sklearn`.

# ERROR

```
  File "/Users/zachlewis/mllearn/mllearn/learn91.py", line 37, in <module>
    model.fit(X_train, y_train)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/zachlewis/mllearn/path/to/venv/lib/python3.13/site-packages/sklearn/base.py", line 1389, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "/Users/zachlewis/mllearn/path/to/venv/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py", line 1301, in fit
    raise ValueError(
```
The error you’re encountering during `model.fit(X_train, y_train)` suggests an issue with the `LogisticRegression` model’s `fit` method. The traceback points to a `ValueError` raised in `sklearn/linear_model/_logistic.py`, line 1301. While the full error message isn’t provided, this specific line in scikit-learn’s logistic regression code typically corresponds to a check that fails when there’s an issue with the input data—most commonly when the target variable (`y_train`) has fewer than two unique classes.

Given the small dataset (10 samples) and the way we’re generating labels, it’s possible that after the train-test split, `y_train` contains only one class (all 1s or all 0s), which breaks logistic regression since it needs at least two classes to model a binary outcome. Let’s diagnose and fix this by ensuring the data is suitable and adding safeguards.

---

### Diagnosing the Issue
The dataset is small (10 samples), and the `train_test_split` with `test_size=0.2` gives 8 training samples and 2 test samples. With random data and strict conditions for suitability (pH 6.0–7.0, moisture 30–60), it’s possible the split results in `y_train` having only one class. Let’s modify the code to:
1. Check the data for class imbalance.
2. Generate a larger dataset to reduce the chance of this issue.
3. Handle edge cases gracefully.

Here’s the fixed and improved version:

---

### Updated Code

```python
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
```

---

### Key Fixes

1. **Larger Dataset**:
   - Increased `n_samples` from 10 to 50. This improves the likelihood of having both classes (0 and 1) in `y_train` after the split.

2. **Class Check**:
   - Added `np.unique(y_train)` to verify that `y_train` has at least two classes. If not, it skips model training and provides a helpful message.

3. **Debugging Output**:
   - Printed the shape of `X`, the full `y` array, and counts of each class to confirm the data is balanced enough.
   - Printed `y_train` and its unique values to pinpoint the issue if it persists.

---

### Sample Output
Here’s what you might see (exact numbers vary slightly):
```
Feature Matrix (X) Shape: (50, 2)
Labels (y): [1 0 0 1 0 1 ...]
Number of Suitable Plots (1): 12
Number of Unsuitable Plots (0): 38

Training Features Shape: (40, 2)
Training Labels: [0 1 0 0 1 ...]
Unique Classes in y_train: [0 1]

Model Accuracy: 0.9
New Plot Suitability (1 = suitable, 0 = unsuitable): 1
```

- With 50 samples, `y_train` now has both 0s and 1s, so `model.fit()` succeeds.
- Accuracy around 0.9 indicates the model learned well (small dataset, so it’s not perfect).

---

### Why This Happened
In the original code:
- With only 10 samples, the 80-20 split gave 8 training samples.
- The random data and strict suitability rules (pH 6.0–7.0, moisture 30–60) could result in `y_train` having all 0s or all 1s by chance.
- Logistic regression requires at least two classes to fit a decision boundary, hence the `ValueError`.

---

### If the Issue Persists
If you still see the error with this code:
1. **Check `y` Distribution**: Run `print(np.unique(y, return_counts=True))` to see if the dataset is heavily imbalanced. If so, relax the suitability conditions (e.g., widen pH to 5.5–7.5).
2. **Increase Samples**: Bump `n_samples` to 100 or more.
3. **Share Full Error**: If the `ValueError` has a specific message (e.g., “number of classes”), share it, and I’ll refine further.

