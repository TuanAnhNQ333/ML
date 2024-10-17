
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error

train_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/train.csv')
test_data = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/test.csv')

print(train_data.head())
print(test_data.head())

print(train_data.isnull().sum())
train_data = train_data.dropna()  

required_columns = ['allelectrons_Total', 'density_Total', 'allelectrons_Average', 'val_e_Average', 
                    'atomicweight_Average', 'ionenergy_Average', 'el_neg_chi_Average', 
                    'R_vdw_element_Average', 'R_cov_element_Average', 'zaratio_Average', 
                    'density_Average']

for col in required_columns:
    if col not in train_data.columns:
        print(f"Column {col} is missing from the training data!")

X = train_data[required_columns]
y = train_data['Hardness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

medae = median_absolute_error(y_test, y_pred)
print(f'Median Absolute Error: {medae}')

for col in required_columns:
    if col not in test_data.columns:
        print(f"Column {col} is missing from the test data!")

X_test_final = test_data[required_columns]

test_predictions = model.predict(X_test_final)

if 'id' in test_data.columns:
    submission = pd.DataFrame({
        'id': test_data['id'],  
        'Hardness': test_predictions
    })
else:
    submission = pd.DataFrame({
        'Hardness': test_predictions
    })

submission.to_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/submission.csv', index=False)