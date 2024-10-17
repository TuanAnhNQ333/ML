# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

# Load data
train_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/train.csv')
test_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/test.csv')

# Preview the data
print(train_data.head())
print(test_data.head())

# Check for missing values
print(train_data.isnull().sum())
train_data = train_data.dropna()  

# Define required columns
required_columns = ['allelectrons_Total', 'density_Total', 'allelectrons_Average', 'val_e_Average', 
                    'atomicweight_Average', 'ionenergy_Average', 'el_neg_chi_Average', 
                    'R_vdw_element_Average', 'R_cov_element_Average', 'zaratio_Average', 
                    'density_Average']

# Check if columns exist in training data
for col in required_columns:
    if col not in train_data.columns:
        print(f"Column {col} is missing from the training data!")

# Prepare features and target
X = train_data[required_columns]
y = train_data['Hardness']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate median absolute error
medae = median_absolute_error(y_test, y_pred)
print(f'Median Absolute Error: {medae}')

# Check if columns exist in test data
for col in required_columns:
    if col not in test_data.columns:
        print(f"Column {col} is missing from the test data!")

# Prepare the test data for prediction
X_test_final = test_data[required_columns]

# Make predictions on the final test data
test_predictions = model.predict(X_test_final)

# Create a submission dataframe
if 'id' in test_data.columns:
    submission = pd.DataFrame({
        'id': test_data['id'],  
        'Hardness': test_predictions
    })
else:
    submission = pd.DataFrame({
        'Hardness': test_predictions
    })

# Save the submission to a CSV file
submission.to_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/submission1.csv', index=False)
