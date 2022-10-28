from http.client import OK
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf


#### dataset 하나로 만들고, 라벨 붙이기
def dataset_extractor():
    folders= os.listdir('lfw')
    img_list, label_list= [], []
    for i in range(len(folders)):
        for j in range(len(os.listdir(os.path.join('lfw', folders[i])))):
            img_list.append(plt.imread(os.path.join('lfw', folders[i], os.listdir(os.path.join('lfw', folders[i]))[j])))
            label_list.append('_'.join(os.listdir(os.path.join('lfw', folders[i]))[j].split('_')[:2]))
        print(i, "th image dataset has been extracted!")

    os.makedirs('dataset', exist_ok=True)

    img_dataset= np.array(img_list)
    label_dataset= np.array(label_list)
    np.save('dataset/img', img_dataset)
    np.save('dataset/label', label_dataset)


### making pairs for training
def create_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    random.seed(2021)
    pairImages, pairLabels= [], []
   
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    numClasses = len(np.unique(labels))
    classes=np.unique(labels)
    idx = [np.where(labels == classes[i]) for i in range(0, numClasses)]
    
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current iteration
        currentImage = images[idxA]
        label = labels[idxA]
        
        # randomly pick an image that belongs to the *same* class
        # label
        posId = random.choice(list(np.where(labels == label)))
        posIdx =random.choice(posId)
        posImage = images[posIdx]
        
        # prepare a positive pair and update the images and labels
        pairImages.append([currentImage, posImage])
        pairLabels.append([1])
        
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negId = random.choice(list(np.where(labels != label)))         
        negIdx =random.choice(negId)
        negImage = images[negIdx]
        
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])
   
    return (np.array(pairImages), np.array(pairLabels))
