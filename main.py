import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.siamese import tf_siamese_nn, euclidean_distance
from src.preprocess import create_pairs
from src.loss import contrastive_loss

# from src.preprocess import dataset_extractor
# image_scale= cv2.IMREAD_COLOR
# dataset_extractor(image_scale)


## gpu boost
## if your PC is equipped with GPU device,tensorflow gpu activation
# cpus= tf.config.experimental.list_physical_devices('CPU')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus= tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

## load datasets
img= np.load('dataset/img.npy')
label= np.load('dataset/label.npy')

### data split
### Because of OOM(Out of Memory) problems, should lessen the size of dataset...
X_train= img[:int(len(img)*0.1)]
X_val= img[int(len(img)*0.1):int(len(img)*0.12)]
y_train= label[:int(len(label)*0.1)]
y_val= label[int(len(label)*0.1):int(len(label)*0.12)]

X_train= X_train / 255.0
X_val= X_val / 255.0

(pairTrain, labelTrain) = create_pairs(X_train, y_train)
(pairTest, labelTest) = create_pairs(X_val, y_val)

IMG_SHAPE= (250, 250, 3)
BASE_OUTPUT= 'output'

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

BATCH_SIZE= 4
EPOCHS= 10

### baseline
es= EarlyStopping(monitor= 'val_loss', mode= 'min', verbose= 1, patience= 5)
model_ckpt= ModelCheckpoint('models/model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
lrop= ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, mode='auto')

def model_baseline():
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # train the model
    history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], \
        validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]), \
        batch_size= BATCH_SIZE, epochs= EPOCHS, callbacks= [es, model_ckpt, lrop])
    return model, history


### 2nd model
opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss=contrastive_loss, optimizer=opt, metrics=["accuracy"], )

history = model.fit([pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:], \
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]), \
    batch_size = BATCH_SIZE, epochs= EPOCHS, verbose= 1, callbacks= [es, model_ckpt, lrop])

## testing the Siamese Neural Network
model.summary()

i= 0
preds= model.predict([np.expand_dims(pairTest[i][0], axis=0), np.expand_dims(pairTest[i][1], axis=0)])
preds > 0.5

labelTest[i]