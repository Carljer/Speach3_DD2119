import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,Conv2DTranspose,BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, initializers

#alldata=np.load('alldata.npz')['data'].item()
alldata=np.load('traindict.npz')['arr_0'].item()
def modelcreator():
    model=Sequential()
    model.add(Dense(256,input_shape=[13],activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(61,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
#a=modelcreator()
#x=np.float32(alldata['lmfcc_train_x'])
#y=alldata['train_y']
def train(data):
    model=modelcreator()
    model.summary()
    epochs=7
    batch_size=256

    x=np.float32(alldata['lmfcc_train_x'])
    y=alldata['train_y']

    yt=np.zeros([y.shape[0],61])
    for i in range(y.shape[0]):
        yt[i,int(y[i])]=1
    y=yt

    model.fit(x,y,batch_size=batch_size,epochs=epochs,verbose=1)
train(alldata)
