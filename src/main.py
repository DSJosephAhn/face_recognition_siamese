import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dense
from src.siamese import tf_siamese_nn, euclidean_distance
from src.preprocess import create_pairs
from src.loss import contrastive_loss

## load datasets
img= np.load('dataset/img.npy')
label= np.load('dataset/label.npy')

### data split
X_train= img[:int(len(img)*0.8)]  # (10586, 250, 250, 3)
X_val= img[int(len(img)*0.8):]   # (2647, 250, 250, 3)
y_train= label[:int(len(label)*0.8)]   #(10586,)
y_val= label[int(len(label)*0.8):]    #(2647,)

IMG_SHAPE= (250, 250, 3)
BATCH_SIZE= 64
EPOCHS= 100
BASE_OUTPUT= 'output'

(pairTrain, labelTrain) = create_pairs(X_train, y_train)
(pairTest, labelTest) = create_pairs(X_val, y_val)

## Build the Siamese Network
img1 = tf.keras.layers.Input(shape=IMG_SHAPE)
img2 =  tf.keras.layers.Input(shape=IMG_SHAPE)
featureExtractor = tf_siamese_nn(IMG_SHAPE)
featsA = featureExtractor(img1)
featsB = featureExtractor(img2)

# finally, construct the siamese network
distance= tf.keras.layers.Lambda(euclidean_distance)([featsA, featsB])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
model = tf.keras.Model(inputs=[img1, img2], outputs=outputs)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# train the model
history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], \
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]), batch_size=1, epochs=100, verbose= 1)


### 2nd model
opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss=contrastive_loss, optimizer=opt,metrics=["accuracy"], )

history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], \
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),batch_size=1, epochs=120, verbose= 1)

## testing the Siamese Neural Network
i= 0
preds = model.predict([pairTest[i][0], pairTest[i][1]])