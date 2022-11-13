import pandas as pd
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras import preprocessing
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, Input, Resizing
from keras.layers.activation.elu import ELU
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

dir_train = 'FMA/mel/mel/training'
dir_val = 'FMA/mel/mel/validation'

# MusicCNN

# Model Training
num_classes = 8
image_height = 128
image_width = 1280
nb_epoch = 50
batch_size = 32

nb_train_samples = 6394
nb_validation_samples = 800

if K.image_data_format() == 'channels_first':
    input_shape = (1, image_height, image_width)
else:
    input_shape = (image_height, image_width, 1)

model = Sequential()

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Testing input layer
model.add(Input(shape=input_shape))
model.add(Resizing(128, 128))

model.add(Conv2D(filters=32, kernel_size=5, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=64, kernel_size=5, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=128, kernel_size=5, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(
    Conv2D(filters=256, kernel_size=3, strides=2, activation='elu', kernel_initializer='glorot_normal', padding='same'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(128, kernel_regularizer=regularizers.l2(0.001)))

model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = RMSprop()

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Image generators
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(image_height, image_width),
    shuffle=True,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    dir_val,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    shuffle=True,
    color_mode='grayscale',
    class_mode='categorical')

# Fit model
model_fit = model.fit(train_generator,
                      steps_per_epoch=(nb_train_samples // batch_size),
                      epochs=nb_epoch,
                      validation_data=validation_generator,
                      validation_steps=(nb_validation_samples // batch_size),
                      # callbacks=[early_stopping],
                      verbose=2)

# Save model
model.save_weights('MusicCNN_50_weights.h5')
model.save('MusicCNN_50.h5')
