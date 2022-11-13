import pandas as pd
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras import preprocessing
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation, Input, Resizing, BatchNormalization
from keras.layers.activation import ELU, relu
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import image
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

dir_train = 'FMA/mel/mel/training'
dir_val = 'FMA/mel/mel/validation'

# Generic CNN (Adiyansjah)

# Model Training
num_classes = 8
image_height = 128
image_width = 1280
nb_epoch = 10
batch_size = 32

nb_train_samples = 6394
nb_validation_samples = 800

if K.image_data_format() == 'channels_first':
    input_shape = (1, image_height, image_width)
else:
    input_shape = (image_height, image_width, 1)

# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model = Sequential()

model.add(Input(shape=input_shape))

model.add(Conv2D(filters=24, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 4), padding='same'))

model.add(Conv2D(filters=48, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 4), padding='same'))

model.add(Conv2D(filters=48, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 4), padding='same'))

model.add(Conv2D(filters=71, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 5), padding='same'))

model.add(Conv2D(filters=95, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', Precision(), Recall()])

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
                      verbose=1)

# Save model
model.save_weights('GenCNN_10_weights.h5')
model.save('GenCNN_10_50.h5')

# Plot accuracy
plt.figure(figsize=(12, 8))
pd.DataFrame(model_fit.history)['val_accuracy'].plot(label='Generic CNN Model')
plt.legend(fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Validation Accuracy (%)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
