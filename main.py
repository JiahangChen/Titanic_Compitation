from tensorflow import keras
import numpy as np
import tensorflow as tf
from data_preload import *


(titanic_train_data, y_label_train, titanic_test_data, y_label_test) = getTrainingData()


x = np.asarray(titanic_train_data.values).astype(np.float32)
y = np.asarray(y_label_train.values).astype(np.float32)
a = np.asarray(titanic_test_data.values).astype(np.float32)
b = np.asarray(y_label_test.values).astype(np.float32)
model = tf.keras.Sequential()
model.add(keras.layers.Dense(units=30,input_dim=7,activation='relu',kernel_initializer='uniform'))
model.add(keras.layers.Dropout(.2,input_shape=(30,)))
model.add(keras.layers.Dense(units=20,kernel_initializer='uniform',activation='relu'))
model.add(keras.layers.Dropout(.2,input_shape=(20,)))
model.add(keras.layers.Dense(units=10,kernel_initializer='uniform',activation='relu'))
model.add(keras.layers.Dropout(.2,input_shape=(10,)))
model.add(keras.layers.Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x,y,epochs=1000,batch_size=50,verbose=0)

l = model.predict(a)

survived = [1 if x > 0.5 else 0 for x in l]

count = 0
for i in range(len(b)):
  if survived[i] == b[i][0]:
    count +=1
print(count/418)
