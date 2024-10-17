import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

# Load dataset
dataset_path = pd.read_csv('/Users/macbook/Documents/egg/sentiment_analysis.csv')

# Text normalization function
def text_normalize(text):
    text = re.sub(r'^RT[\s]+', '', text)  # Remove retweets
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    return text_tokens

# Get word frequencies function
def get_freqs(df):
    freqs = defaultdict(lambda: 0)
    for idx, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        tokens = text_normalize(tweet)
        for token in tokens:
            pair = (token, label)
            freqs[pair] += 1
    return freqs

# Feature extraction function
def get_feature(text, freqs):
    tokens = text_normalize(text)
    X = np.zeros(3)  # We use 3 features
    X[0] = 1  # Bias term
    for token in tokens:
        X[1] += freqs[(token, 0)]  # Count for negative sentiment
        X[2] += freqs[(token, 1)]  # Count for positive sentiment
    return X

# Prepare data
X = []
y = []

freqs = get_freqs(dataset_path)
for idx, row in dataset_path.iterrows():
    tweet = row['tweet']
    label = row['label']
    X_i = get_feature(tweet, freqs)
    X.append(X_i)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split the dataset into training, validation, and test sets
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state, shuffle=is_shuffle)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, shuffle=is_shuffle)

# Normalize the data
normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute loss function
def compute_loss(y_hat, y):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)  # Avoid division by zero
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

# Predict function
def predict(X, theta):
    dot_product = np.dot(X, theta)
    y_hat = sigmoid(dot_product)
    return y_hat

# Compute gradient function
def compute_gradient(X, y, y_hat):
    return np.dot(X.T, (y_hat - y)) / y.size

# Update theta function
def update_theta(theta, gradient, lr):
    return theta - lr * gradient

# Compute accuracy function
def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta).round()
    acc = (y_hat == y).mean()
    return acc

# Initialize parameters for training
lr = 0.01
epochs = 200
batch_size = 128

np.random.seed(random_state)
theta = np.random.uniform(size=X_train.shape[1])

# Training loop
for epoch in range(epochs):
    y_hat_train = predict(X_train, theta)
    loss = compute_loss(y_hat_train, y_train)
    
    gradient = compute_gradient(X_train, y_train, y_hat_train)
    theta = update_theta(theta, gradient, lr)
    
    if epoch % 10 == 0:  # Print progress every 10 epochs
        train_acc = compute_accuracy(X_train, y_train, theta)
        val_acc = compute_accuracy(X_val, y_val, theta)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}")

# Final evaluation on test set
test_acc = compute_accuracy(X_test, y_test, theta)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
