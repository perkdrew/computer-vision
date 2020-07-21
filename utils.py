import os
import random
import numpy as np

import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Lambda
from keras import backend as K

# metrics
def euclidean_distance(vectors):
    v1, v2 = vectors
    sum_square = K.sum(K.square(v1-v2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, D):
    margin = 1
    return K.mean(y_true*K.square(D)+(1 - y_true)*K.maximum((margin-D),0))

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_pairs(X, y, num_classes):
    '''helper function used to generate
    negative and positive images for training'''
    pairs, labels = [], []
    # index of images in X and y for each class
    class_idx = [np.where(y==i)[0] for i in range(num_classes)]
    # the min number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
    
    for c in range(num_classes):
        for n in range(min_images):
            # create positive pair
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[c][n+1]]
            pairs.append((img1,img2))
            labels.append(1)
            
            # create negative pair
            neg_list = list(range(num_classes))
            neg_list.remove(c)
            # select a random class from the negative list
            # this class will be used to form the negative pair
            neg_c = random.sample(neg_list,1)[0]
            img1 = X[class_idx[c][n]]
            img2 = X[class_idx[neg_c][n]]
            pairs.append((img1,img2))
            labels.append(0)

    return np.array(pairs), np.array(labels)


def create_siamese_nn(input_shape):
    model = Sequential(name='Conv_Network')
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model


def get_data(dir)
    X_train, y_train = [], []
    X_test, y_test = [], []
    subfolders = sorted([f.path for f in os.scandir(faces_dir) if f.is_dir()])
    # iterate through the list of subfolders (subjects)
    for idx, folder in enumerate(subfolders):
        for file in sorted(os.listdir(folder)):
            img = load_img(folder+'/'+file, color_mode='grayscale')
            img = img_to_array(img).astype('float32')/255
            if idx < 35:
                X_train.append(img)
                y_train.append(idx)
            else:
                X_test.append(img)
                y_test.append(idx-35)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return (X_train, y_train), (X_test, y_test)
