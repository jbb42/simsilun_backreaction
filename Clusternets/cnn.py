import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import numpy as np
import scipy
import h5py as h5
import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from subprocess import check_output

config = cp.ConfigParser()
config.read("config.ini")
sub_box = int(config.get("myvars", "sub_box"))


def model(in_shape=(sub_box,sub_box,sub_box,1)):
    """Build a 3D convolutional neural network model."""
    model = Sequential()
    
    model.add(Conv3D(8, 7,  strides=2, padding='same', activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),input_shape=in_shape))
    #model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv3D(16, 7,  strides=2, padding='same', activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))

    model.add(Conv3D(32, 7,  strides=2, padding='same', activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
   # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    
    model.add(Conv3D(16, 7,  strides=2, padding='same', activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Conv3D(8, 7,  strides=2, padding='same', activation="relu",kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    #model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation="sigmoid"))

    # Define the model.
    return model



