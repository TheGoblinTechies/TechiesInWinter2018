import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

##################################################
#Warning: If you want to use this code in any other course,
#a necessary reference should be noted in the beginning.
#By: TheGoblinTechies at 2017.09.30
##################################################

#read in data
traindata = np.loadtxt("dataset3.txt",delimiter=",").astype(np.float)
testdata = np.loadtxt("dataset4.txt",delimiter=",").astype(np.float)

#create matrixes for read-in data
x_train = np.ones((954,10),dtype = float)
y_train = np.ones((954,1),dtype = float)
x_test = np.ones((328,10),dtype = float)
y_test = np.ones((328,1),dtype = float)

for i in range(0,954):
    y_train[i,0]=traindata[i,-1];
    x_train[i,]=traindata[i,0:-1];

for i in range(0,328):
    y_test[i,0]=testdata[i,-1];
    x_test[i,]=testdata[i,0:-1];

#binary for loss function
from keras.utils.np_utils import to_categorical
y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)

print x_train

#fully connected layer
#This can be DIYed for your favor
#Eg:add layers, change the number of units
#	or the activation function
#	or the loss function, optimizer
#
#keras's web for specific function 
#https://keras.io/
##################################################
model = Sequential()
model.add(Dense(units=10, input_dim=10))
model.add(Activation('sigmoid'))
model.add(Dense(units=6))
model.add(Activation('sigmoid'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_binary_train, validation_data= (x_test, y_binary_test), epochs=10, batch_size=3)
##################################################



# list all data in history

print(history.history.keys())
# summarize history for accuracy while trainning
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss while trainning
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print (model.test_on_batch(x_test, y_binary_test))
