# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error
from sklearn.decomposition import PCA

# Load data
train_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/train.csv')
test_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/test.csv')

# Preview the data
print(train_data.head())
print(test_data.head())

# Check for missing values and drop rows with missing target values (Hardness)
print(train_data.isnull().sum())
train_data = train_data.dropna(subset=['Hardness'])  # Only drop rows missing the target

# Define required columns for training
required_columns = ['allelectrons_Total', 'density_Total', 'allelectrons_Average', 'val_e_Average', 
                    'atomicweight_Average', 'ionenergy_Average', 'el_neg_chi_Average', 
                    'R_vdw_element_Average', 'R_cov_element_Average', 'zaratio_Average', 
                    'density_Average']

# Check for missing columns
missing_columns_train = [col for col in required_columns if col not in train_data.columns]
missing_columns_test = [col for col in required_columns if col not in test_data.columns]

if missing_columns_train:
    raise ValueError(f"Missing columns in training data: {missing_columns_train}")
if missing_columns_test:
    raise ValueError(f"Missing columns in test data: {missing_columns_test}")

# Prepare features and target
X = train_data[required_columns]
y = train_data['Hardness']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: Apply PCA for dimensionality reduction (faster training, might lose some precision)
pca = PCA(n_components=8)  # Reduce to 8 principal components (you can tune this number)
X_pca = pca.fit_transform(X_scaled)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize and train the model (use a more powerful regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate median absolute error
medae = median_absolute_error(y_test, y_pred)
print(f'Median Absolute Error: {medae}')

# Prepare the test data
X_test_final = test_data[required_columns]

# Scale and apply PCA to test data
X_test_final_scaled = scaler.transform(X_test_final)
X_test_final_pca = pca.transform(X_test_final_scaled)

# Make predictions on the final test data
test_predictions = model.predict(X_test_final_pca)

# Create a submission dataframe
submission = pd.DataFrame({
    'id': test_data['id'],  
    'Hardness': test_predictions
})

# Save the submission to a CSV file
submission.to_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/submission.csv', index=False)
