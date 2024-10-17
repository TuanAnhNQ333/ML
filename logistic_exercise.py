import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset_path = pd.read_csv('/Users/macbook/Documents/egg/egg-linear-regression-contest/titanic_modified_dataset.csv')
print(dataset_path.head())
df = pd.read_csv(
    dataset_path,
    index_col = 'PassengerId'
)




# ham sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def predict(X, theta):
    dot_product = np.dot(X, theta)
    y_hat = sigmoid(dot_product)
    
    return y_hat
# ham tinh loss
def compute_loss(y_hat, y):
    y_hat = np.clip(
        y_hat, 1e-7, 1 - 1e-7
    )
    
    return (
        -y * \
        np.log(y_hat) - (1 - y) * \
        np.log(1 - y_hat)
    ).mean()
# ham tinh gradient
def compute_gradient(X, y, y_hat):
        return np.dot(
            X.T, (y_hat - y) 
        ) / y.size
# ham cap nhat trong so
def update_theta(theta, gradient, lr):
    return theta - lr * gradient
# ham tinh do chinh xac
def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    acc = (y_hat == y).mean()
    
    return acc
train_accs = []
train_losses = []
val_accs = []
val_losses = []

for epoch in range(epochs):
    train_batch_losses = []
    train_batch_accs = []
    val_batch_losses = []
    val_batch_accs = []
    
    for i in range(0. X_train.shape[0], batch_size) :
        X_i = X_train[i:i+batch_size]
        Y_i = y_train[1:i+batch_size]
        
        y_hat = predict(X_i, theta)
        
        train_loss = compute_loss(y_hat, y_i)
        gradien = compute_gradient(X_i, y_i, y_hat)
        
        theta = update_theta(theta, gradient, lr)
        
        train_batch_losses.append(train_loss)
        
        train_accs