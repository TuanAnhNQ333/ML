# import libraries
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

dataset_path = pd.read_csv('/Users/macbook/Documents/egg/sentiment_analysis.csv')

required_columns = ['id']

def text_normalize(text):
    #retweet old acronym "RT" removal
    text = re.sub(r'^RT[\s]+', '', text)
    #hyperlinks removal
    text = re.sub(r'#', '', text)
    #punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    #tokenization
    tokenizer = TweetTokenizer(
        preserve_case = False,
        strip_handles = True,
        reduce_len = True
    )
    text_tokens = tokenizer.tokenize(text)
    
    return text_tokens
def get_freqs(dataset_path):
    freqs = defaultdict(lambda : 0)
    for idx, row in dataset_path.iterrows():
        tweet = row['tweet']
        label = row['label']
        
        tokens = text_normalize(tweet)
        for token in tokens:
            pair = (token, label)
            freqs[pair] += 1
    return freqs
def get_feature(text, freqs):
    tokens = text_normalize(text)
    
    X = np.zeros(3) 
    X[0] = 1
    
    for token in tokens:
        X[1] += freqs[(token, 0)] 
        X[2] += freqs[(token, 1)]
        
    return X

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

val_size = 0.2 
test_size = 0.125 
random_state = 2 
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split( 
        X, y,
    test_size=val_size , 
    random_state=random_state , 
    shuffle=is_shuffle
)
X_train, X_test, y_train, y_test = train_test_split( 
    X_train , 
    y_train ,
    test_size=test_size ,
    random_state=random_state ,
    shuffle=is_shuffle 
)
normalizer = StandardScaler()
X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:]) 
X_val[:, 1:] = normalizer.transform(X_val[:, 1:]) 
X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_loss(y_hat, y): 
    y_hat = np.clip(
        y_hat, 1e-7, 1 - 1e-7 
)
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()

def predict(X, theta):
    dot_product = np.dot(X, theta) 
    y_hat = sigmoid(dot_product)
    
    return y_hat
def compute_gradient(X, y, y_hat): 
    return np.dot(
        X.T, (y_hat - y) 
    ) / y.size
def update_theta(theta, gradient, lr): 
    return theta - lr * gradient

def compute_accuracy(X, y, theta): 
    y_hat = predict(X, theta).round()
    acc = (y_hat == y).mean() 
    
    return acc
lr = 0.01
epochs = 200 
batch_size = 128

np.random.seed(random_state) 
theta = np.random.uniform(
    size=X_train.shape[1] 
)

val_set_acc = compute_accuracy(X_val, y_val, theta) 
test_set_acc = compute_accuracy(X_test , y_test , theta) 
print('Evaluation on validation and test set:') 
print(f'Accuracy: {val_set_acc}')
print(f'Accuracy: {test_set_acc}')











