import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
X = [[22.3, -1.5, 1.1, 1]]
theta = [0.1, -0.15, 0.3, -0.2]
def predict(X, theta):
    z = np.dot(X, theta)
    
    return 1 / (1 + np.exp(-z))
print(predict(X, theta))


y = np.array([1,0,0,1])
y_hat = np.array([0.8, 0.75, 0.3, 0.95])
    
def compute_loss(y_hat, y):
    y_hat = np.clip(
        y_hat, 1e-7, 1 - 1e-7
    )
    return (-y * np.log(y_hat) - ( 1 - y ) * np.log(1 - y_hat)).mean()
print(compute_loss(y_hat, y))




