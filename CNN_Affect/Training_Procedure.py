import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, GaussianDropout, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential, load_model
import tensorflow as tf
# import autokeras as ak
import timeit
# import gc


class Training_Procedure:

    def __init__(self, model, image_size, train_input_path, train_output, test_input_path, test_output, batch_size, epochs, regression=True):

        self.model = model
        self.image_size = image_size

        self.train_input_path = train_input_path
        self.train_output = train_output

        self.test_input_path = test_input_path
        self.test_output = test_output

        self.batch_size = batch_size
        self.epochs = epochs

        self.regression = regression

    # have yet known how to load data for regression
    #   since the label values must be type integers

    def load_train_data(self, sub_set="training", val_split=0.2):
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_input_path,
            labels=list(self.train_output.astype(int)),
            label_mode='categorical',
            color_mode='rgb',
            # Use 20% data as testing data.
            validation_split=val_split,
            subset=sub_set,
            # Set seed to ensure the same split when loading testing data.
            seed=123,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size
        )
        return train_data.map(lambda x, y: (x/255., y))

    def load_test_data(self):
        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_input_path,
            labels=list(self.test_output.astype(int)),
            label_mode='categorical',
            color_mode='rgb',
            # Set seed to ensure the same split when loading testing data.
            seed=123,
            image_size=(self.image_size, self.image_size),
            batch_size=self.batch_size
        )
        return test_data.map(lambda x, y: (x/255., y))

    # Training function
    def training(self):
        train_data = self.load_train_data()
        val_data = self.load_train_data(sub_set="validation")
        # Training process
        startTime = timeit.default_timer()
        hist = self.model.fit(train_data,
                              validation_data=val_data,
                              epochs=self.epochs,
                              batch_size=self.batch_size, verbose=1)
        print('Training time:', timeit.default_timer() - startTime)

        return hist

    # Testing function
    def testing(self):
        # Testing
        test_data = self.load_test_data()
        startTime = timeit.default_timer()
        testLoss, testAccuracy = self.model.evaluate(test_data)
        print('Testing time:', timeit.default_timer() - startTime)

        return testLoss, testAccuracy

    def process_classification(self):

        # Training
        hist = self.training()

        # Testing
        testLoss, testAccuracy = self.testing()

        return hist.history, testLoss, testAccuracy, self.model.count_params()

    def process_regression(self):
        pass

    def process(self):
        if (not self.regression):
            return self.process_classification()
        return self.process_regression()
