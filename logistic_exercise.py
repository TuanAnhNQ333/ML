import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataset_path = 'titanic_modified_dataset.csv'
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