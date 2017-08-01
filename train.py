import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

np.random.seed(1234)

def buildModel():
  model = Sequential()
  layers = [1, 50, 100, 5]

  model.add(LSTM(
    layers[1],
    input_shape=(None, 1),
    return_sequences=True
  ))
  model.add(Dropout(0.2))

  model.add(LSTM(
    layers[2],
    return_sequences=False
  ))
  model.add(Dropout(0.2))

  model.add(Dense(layers[3]))
  model.add(Activation('linear'))

  start = time.time()
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  print (f"Compilation Time : {time.time() - start}")

  return model