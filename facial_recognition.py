import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Lambda
from keras import backend as K

faces_dir = 'att_faces/'

X_train, y_train = [], []
X_test, y_test = [], []

# get list of subfolders from faces_dir
# each subfolder contains images from one subject
subfolders = sorted([f.path for f in os.scandir(faces_dir) if f.is_dir()])

# iterate through the list of subfolders (subjects)
# idx is the subject ID
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

'''
subject_idx = 4
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6),
      (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(10,10))
subject_img_idx = np.where(y_train==subject_idx)[0].tolist()

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = X_train[subject_img_idx[i]]
    img = np.squeeze(img)
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()


# plot the first 9 subjects
subjects = range(10)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6),
      (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(10,12))
subject_img_idx = [np.where(y_train==i)[0].tolist()[0] for i in subjects]

for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = X_train[subject_img_idx[i]]
    img = np.squeeze(img)
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Subject {}'.format(i))
plt.show()
plt.tight_layout()
'''

def siamese_nn(input_shape):
    model = Sequential(name='Conv_Network')
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model

input_shape = X_train.shape[1:]
siamese_nn = siamese_nn(input_shape)

input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)

output_top = siamese_nn(input_top)
output_bottom = siamese_nn(input_bottom)

def euclidean_distance(vectors):
    v1, v2 = vectors
    sum_square = K.sum(K.square(v1-v2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

distance = Lambda(euclidean_distance, output_shape=(1,))([output_top,
                  output_bottom])

model = Model(inputs=[input_top, input_bottom], outputs=distance)
print(model.summary())

