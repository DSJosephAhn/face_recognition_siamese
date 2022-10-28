import tensorflow as tf
import keras.backend as K

def contrastive_loss(y, preds, margin=1):
 # explicitly cast the true class label data type to the predicted
 # class label data type 
 y= tf.cast(y, preds.dtype)
 # calculate the contrastive loss between the true labels and
 # the predicted labels
 squaredPreds= K.square(preds)
 squaredMargin= K.square(K.maximum(margin - preds, 0))
 loss= 1-K.mean(y * squaredPreds + (1 - y) * squaredMargin) 
 return loss