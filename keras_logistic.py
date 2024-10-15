import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses
from keras import optimizers

# 1. Prepare data 
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# 2. Build model 
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation('sigmoid'))

# 3. gradient descent optimizer and loss function 
sgd = optimizers.SGD(lr=0.05)
model.compile(loss=losses.binary_crossentropy, optimizer=sgd)

# 4. Train the model 
model.fit(X, y, epochs=3000, batch_size=1) 
# 1. prepare data 
from __future__ import print_function 
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('x_train shape:\t', x_train.shape)
print('x_test shape:\t', x_test.shape)
print('y_train shape:\t', y_train.shape)
print('y_test shape:\t', y_test.shape)
# data normalization
x_train = x_train/255.
x_test = x_test/255. 
num_classes = 10 
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import metrics 
# 2. buid model 
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 3. loss, metrics 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])
from keras import metrics 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])






