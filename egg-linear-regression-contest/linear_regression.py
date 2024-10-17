# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

# Load the dataset
train_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/train.csv')
test_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/test.csv')

# Check the data structure
print(train_data.head())
print(test_data.head())

# Check for missing values and handle them if necessary (e.g., by filling or dropping)
print(train_data.isnull().sum())
train_data = train_data.dropna()  # Dropping rows with missing values (or you can use fillna)

# Features (X) and target (y)
# Ensure the columns are present in the data, and modify if needed
required_columns = ['num_electrons', 'valence_electrons', 'atomic_number', 'electronegativity',
                    'covalent_radius', 'vdw_radius', 'ionization_energy']
for col in required_columns:
    if col not in train_data.columns:
        print(f"Column {col} is missing from the training data!")

X = train_data[required_columns]
y = train_data['Hardness']

# Split the data into training and test sets (using the training set here for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation (test) set
y_pred = model.predict(X_test)

# Calculate Median Absolute Error (MedAE)
medae = median_absolute_error(y_test, y_pred)
print(f'Median Absolute Error: {medae}')

# Prepare predictions for the actual test set (from test.csv)
# Check if test_data contains the required columns
for col in required_columns:
    if col not in test_data.columns:
        print(f"Column {col} is missing from the test data!")

X_test_final = test_data[required_columns]

# Predict the Hardness for the test data
test_predictions = model.predict(X_test_final)

# Prepare submission file (handle the case where 'id' might not exist)
if 'id' in test_data.columns:
    submission = pd.DataFrame({
        'id': test_data['id'],  # assuming 'id' column is present in the test set
        'Hardness': test_predictions
    })
else:
    submission = pd.DataFrame({
        'Hardness': test_predictions
    })

# Save the submission to CSV
submission.to_csv('submission.csv', index=False)
