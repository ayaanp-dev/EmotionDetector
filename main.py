import tensorflow as tf
from models import rnn
import numpy as np

tf.config.list_physical_devices('GPU')

x_train = np.load("x_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
x_test = np.load("x_test.npy", allow_pickle=True)
y_test = np.load("y_test.npy", allow_pickle=True)

print(x_train, y_train, x_test, y_test)

model = rnn(x_train.shape[1:])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))