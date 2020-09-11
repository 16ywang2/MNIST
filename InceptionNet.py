from mlxtend.data import mnist_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras
from sklearn import model_selection
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Activation, Flatten, Concatenate, Dropout
from keras.models import Model

# load in the data
X_orig, y_orig = mnist_data()
# print(X[1])
image_width = image_height = int(math.sqrt(len(X_orig[0])))

# reshape X
X_orig = np.array(X_orig)
X = X_orig.reshape([X_orig.shape[0], image_height, image_height, 1])

# print(X_orig[0])
# print(X.shape)

# split into training and test set
X_train, X_test, y_train_orig, y_test_orig = model_selection.train_test_split(X, y_orig, test_size=0.02,
                                                                              random_state=17)


def plot_image(vector):
    """
    A function to reshape and plot an image from the original flattened vector
    :param vector
    :return: a plot of the vector
    """
    image = vector.reshape(image_width, image_height)
    plt.axis("off")
    plt.imshow(image, cmap="Greys")
    plt.show()


def convert_to_onehot(y_vec):
    """
    A function that reshapes the y vector (1,2,5...) into one hot tensor[[0,1,0,0,0,0,0...], [0,0,1,0,0,0,0...]]
    :param y_vec
    :return: one_hot tensor
    """
    classes = len(set(y_vec))
    classes_tf = tf.constant(classes)
    onehot = tf.one_hot(y_vec, depth=classes_tf, axis=1)
    return onehot


# print(X_train.shape)
# print(y_train[0])
# print(plot_image(X_train[0]))


# convert y to one-hot vectors
y_train = convert_to_onehot(y_train_orig)
y_test = convert_to_onehot(y_test_orig)


def model(input_shape):
    """
    Defining a model that uses the InceptionNet architecture
    :param input_shape: The shape of the image
    :return: The model
    """
    X_input = Input(input_shape)

    # first block
    X_Conv3 = Conv2D(filters=128, kernel_size=[3, 3], padding="SAME")(X_input)
    X_Conv5 = Conv2D(filters=64, kernel_size=[5, 5], padding="SAME")(X_input)
    X_Max = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding="SAME")(X_input)
    X = Concatenate(axis=-1)([X_Conv3, X_Conv5, X_Max])

    # second block
    X_Conv3_11 = Conv2D(filters=32, kernel_size=[1, 1], padding="SAME")(X)
    X_Conv5_11 = Conv2D(filters=32, kernel_size=[1, 1], padding="SAME")(X)
    X_Conv3 = Conv2D(filters=128, kernel_size=[3, 3], padding="SAME")(X_Conv3_11)
    X_Conv5 = Conv2D(filters=64, kernel_size=[5, 5], padding="SAME")(X_Conv5_11)
    X_Max = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same")(X)
    X_Max_11 = Conv2D(filters=32, kernel_size=[1, 1], padding="SAME")(X_Max)
    X = Concatenate(axis=-1)([X_Conv3, X_Conv5, X_Max_11])

    # fully connected layers
    X = Flatten()(X)
    X = Dense(units=128, activation="relu", kernel_initializer=keras.initializers.glorot_normal)(X)
    X = Dense(units=128, activation="relu", kernel_initializer=keras.initializers.glorot_normal)(X)
    X = Dense(units=10, activation="softmax")(X)

    Inception = Model(inputs=X_input, outputs=X)
    return Inception


# Initialize and compile model
InceptionModel = model(X_train.shape[1:])
InceptionModel.compile(optimizer="Adam", loss="CategoricalCrossentropy", metrics =["accuracy"])

# train and evaluate
InceptionModel.fit(x=X_train, y=y_train, batch_size=64, epochs=8, verbose=2)
InceptionModel.evaluate(x=X_test, y=y_test, batch_size=64)