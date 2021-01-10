# Import modules

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2
import numpy as np


class Sharpen(tf.keras.layers.Layer):
    """
    Sharpen layer sharpens the edges of the image.
    """
    
    def __init__(self, num_outputs) :
        super(Sharpen, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape) :
        self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.cast(self.kernel, tf.float32)

    def call(self, input_) :
        return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        
        
class TrafficClassifier:
    def createCNN(width, height, depth, classes):
        model =  Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        model.add(Input(shape=(inputShape), name="input_layer", dtype='float32'))
        #Sharpen Layer to sharpen the edges of the image.
        model.add(Sharpen(num_outputs=inputShape))
        
        # CONV => RELU => BN => POOL
        model.add(Conv2D(64, (3, 3), padding="same", input_shape= inputShape, 
                         kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
       
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model