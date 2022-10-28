import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dense


def tf_siamese_nn(shape, embedding= 64, fineTune= False):
    inputs = tf.keras.layers.Input(shape)
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=shape, include_top=False, weights='imagenet')
    
    if fineTune == False:
        base_model.trainable= False
    else:
        base_model.trainable= True
        # Fine-tune from this layer onwards
        fine_tune_at= len(base_model.layers) - int(len(base_model.layers)*.10)
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in (base_model.layers[:fine_tune_at]):
          layer.trainable =  False

    x= base_model(inputs)
    x= tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs= tf.keras.layers.Dense(embedding)(x)
    model= tf.keras.Model(inputs, outputs)
    return model

def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


