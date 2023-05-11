import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from keras.optimizers import Adam

train = pd.read_csv("Train.csv").values
X_train = train[:, 1:]
Y_train = train[:, 0]

test = pd.read_csv("Test.csv").values
X_test = test[:, 1:]
Y_test = test[:, 0]

X_train, X_test = (X_train / 255.0), (X_test / 255.0)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(784,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

opt = Adam(learning_rate=0.001)

model.compile(optimizer=opt, loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])

bs = 64
n_epoch = 10

model.fit(X_train, Y_train, batch_size=bs, epochs=n_epoch, validation_data=(X_test, Y_test))

pdc = model.predict(X_test)

for real, predicted in zip(Y_test, pdc):
    max_index = np.argmax(predicted)
    print("Значення {} було передбачено як {}".format(real, max_index))
