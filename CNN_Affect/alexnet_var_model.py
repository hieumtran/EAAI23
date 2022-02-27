import gc
import timeit
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GaussianDropout, GlobalAveragePooling2D, MaxPooling2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# from tensorflow.keras.utils import image_dataset_from_directory
# import autokeras as ak


# arguments:
#   image_size: assuming all the input photo has size m*m. The default of this dataset is 128
#   R_or_C: regression (0) or classification (1). We are more interested in regression so the default is set to 0
def alexnet_var_model(image_size=128,
                      regression=True,
                      conv_shapes=[[16, (9, 9)], [32, (7, 7)], [64, (5, 5)], [
                          128, (3, 3)], [128, (3, 3)]],
                      dropout=[0.2, 0.5]):

    model = Sequential()

    # Convolution blocks
    for i in range(len(conv_shapes)):
        if (i == 0):
            # the package requires input_shape when Conv2D is the first layer of the model
            model.add(Conv2D(conv_shapes[i][0], conv_shapes[i][1], input_shape=(
                image_size, image_size, 3), padding='same', use_bias=False))
        else:
            model.add(
                Conv2D(conv_shapes[i][0], conv_shapes[i][1], padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(GaussianDropout(dropout[0]))

    # Flatten
    model.add(Flatten())

    # Adding 2 Dense layers
    for i in range(2):
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(GaussianDropout(dropout[1]))

    # Output layer
    if (regression):      # 0 represents Regression, 1 represents Classification
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error',
                      metrics=['accuracy'])
    else:
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',  metrics=['accuracy'])

    return model
