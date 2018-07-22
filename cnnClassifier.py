from keras import regularizers
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout, Conv2D,BatchNormalization , UpSampling2D as US2D
from keras.callbacks import EarlyStopping 
import pandas as pd

import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))

classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))


classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))
# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu',strides=2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu',strides=2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Add a third layer
#classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#add a fourth layer
#classifier.add(Conv2D(512 , (3, 3), activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
#classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(256, input_dim=256,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
classifier.add(Dense(units = 30, activation = 'softmax'))
metric=['accuracy']
# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
rotation_range=30)

training_set = train_datagen.flow_from_directory('./HE_Chal1',
                                                 target_size = (256, 256),
                                                 batch_size = 8,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./Validation_1',
                                            target_size = (256,256),
                                            batch_size = 8,
                                            class_mode = 'categorical')
early_stops = EarlyStopping(patience=3, monitor='val_acc')
classifier.fit_generator(training_set,                 steps_per_epoch = 100,epochs = 50,validation_data = test_set,validation_steps = 100)	
classifier.save_weights('classifier.h5')
classifier.summary()
