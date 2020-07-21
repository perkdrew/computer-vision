import utils
import numpy as np

from keras.models import Model
from keras.layers import Input, Lambda

faces_dir = 'att_faces/'

# import training and test data
(X_train, y_train), (X_test, y_test) = utils.get_data(faces_dir)
num_classes = len(np.unique(y_train))

# create siamese neural network
input_shape = X_train.shape[1:]
siamese_nn = utils.create_siamese_nn(input_shape)
input_left = Input(shape=input_shape)
input_right = Input(shape=input_shape)
output_left = siamese_nn(input_left)
output_right = siamese_nn(input_right)
distance = Lambda(utils.euclidean_distance, output_shape=(1,))([output_left, output_right])
model = Model(inputs=[input_left, input_right], outputs=distance)

# train model
training_pairs, training_labels = utils.create_pairs(X_train, y_train, 
                                                     num_classes=num_classes)
model.compile(loss=utils.contrastive_loss, optimizer='adam', 
              metrics=[utils.accuracy])
model.fit([training_pairs[:,0], training_pairs[:,1]], training_labels,
           batch_size=128, epochs=10)

# save model
model.save('siamese_nn.h5')

