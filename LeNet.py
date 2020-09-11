from mlxtend.data import mnist_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import keras
from sklearn import model_selection
from keras.layers import Dense, Input, Conv2D, MaxPool2D, Activation, Flatten
from keras.models import Model

# load in the data
X_orig, y_orig = mnist_data()
# print(X[1])
image_width = image_height = int(math.sqrt(len(X_orig[0])))

# reshape X
X_orig = np.array(X_orig)
X = X_orig.reshape([X_orig.shape[0], image_height, image_height, 1])

#print(X_orig[0])
#print(X.shape)

# split into training and test set
X_train, X_test, y_train_orig, y_test_orig = model_selection.train_test_split(X, y_orig, test_size=0.02, random_state=17)


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


# LeNet5
def model(input_shape):
    """
    Defining the model using LeNet 5 structure
    :param input_shape: the shape of the image
    :return: the model to be trained
    """
    X_input = Input(input_shape)

    # first layer
    X = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', name="conv1")(X_input)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1")(X)
    #print(X)

    # second layer
    X = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', name="conv2")(X)
    X = Activation("relu")(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2")(X)
    #print(X)

    # fully connected layer
    X = Flatten()(X)
    X = Dense(120, activation= "relu", kernel_initializer= keras.initializers.glorot_normal)(X)
    X = Dense(84,activation= "relu", kernel_initializer= keras.initializers.glorot_normal)(X)
    X = Dense(10, activation= "softmax")(X)

    # Create the model instance
    model = Model(inputs=X_input, outputs=X)
    return model


# initialize the model
LeNet = model(X_train.shape[1:])

# compile model
LeNet.compile(optimizer="Adam", loss="CategoricalCrossentropy", metrics=["accuracy"])

# fit the model
LeNet.fit(x=X_train, y=y_train, batch_size=32, epochs=15, verbose=2)

# evaluate the model
pred = LeNet.evaluate(x=X_test, y=y_test, batch_size=32)

# make predictions
predictions_prob = LeNet.predict(X_test, batch_size=32)
predictions = predictions_prob.argmax(axis=1)
comparison = pd.concat([pd.Series(y_test_orig, name="Actual Value"), pd.Series(predictions, name="Predictions")], axis=1)




