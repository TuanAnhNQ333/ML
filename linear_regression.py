# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check the data to understand the structure
print(train_data.head())

# Assuming the dataset contains features like 'num_electrons', 'valence_electrons', etc.
# You will need to replace these with the actual column names in your dataset

# Features (X) and target (y)
X = train_data[['num_electrons', 'valence_electrons', 'atomic_number', 'electronegativity',
                'covalent_radius', 'vdw_radius', 'ionization_energy']]
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
X_test_final = test_data[['num_electrons', 'valence_electrons', 'atomic_number', 'electronegativity',
                          'covalent_radius', 'vdw_radius', 'ionization_energy']]

# Predict the Hardness for the test data
test_predictions = model.predict(X_test_final)

# Prepare submission file (replace 'test.csv' with actual test data file)
submission = pd.DataFrame({
    'id': test_data['id'],  # assuming 'id' column is present in the test set
    'Hardness': test_predictions
})

# Save the submission to CSV
submission.to_csv('submission.csv', index=False)
