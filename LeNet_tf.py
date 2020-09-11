from mlxtend.data import mnist_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import ceil
import tensorflow as tf
from sklearn import model_selection

# load in the data
X_orig, y_orig = mnist_data()
# print(X[1])
image_width = image_height = int(math.sqrt(len(X_orig[0])))

# reshape X
X_orig = np.array(X_orig).astype(float)
X = X_orig.reshape([X_orig.shape[0], image_height, image_height, 1])

# form train & test set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y_orig, test_size=0.02, random_state=17)


# change y to one_hot vector
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


# form the dataset in tf
train_tf = tf.data.Dataset.from_tensor_slices((X_train, convert_to_onehot(Y_train)))
batches = train_tf.batch(batch_size=32, drop_remainder=True)


# define the different layers
def conv2d(input, filters, stride):
    out = tf.nn.conv2d(input, filters, strides=[1, stride, stride, 1], padding="VALID")
    return tf.nn.relu(out)


def maxpool2d(input, stride):
    return tf.nn.max_pool2d(input, ksize=2, strides=[1, stride, stride, 1], padding="VALID")


def dense(input, filters, activation):
    if activation == "relu":
        out = tf.matmul(input, filters)
        return tf.nn.relu(out)
    if activation == "softmax":
        out = tf.matmul(input, filters)
        return tf.nn.softmax(out)


# Initialize weights
def initialize(input_shape, name):
    initializer = tf.initializers.glorot_uniform()
    return tf.Variable(initializer(input_shape), trainable=True, name=name, dtype=tf.float32)


shapeList = [
    [5, 5, 1, 6],
    [5, 5, 6, 16],
    [256, 120],
    [120, 84],
    [84, 10]
]

weights = []
for i in range(len(shapeList)):
    weight = initialize(shapeList[i], f'weights_{i}')
    weights.append(weight)


# Set up the model, it should correspond to the one built in Keras
def model(input, weightList):
    c1 = conv2d(input, weightList[0], 1)
    m1 = maxpool2d(c1, 2)
    c2 = conv2d(m1, weightList[1], 1)
    m2 = maxpool2d(c2, 2)
    flattened = tf.reshape(m2, shape=[m2.shape[0], -1])
    d1 = dense(flattened, weightList[2], "relu")
    d2 = dense(d1, weightList[3], "relu")
    d3 = dense(d2, weightList[4], "softmax")
    return d3


# define the loss function
def loss(model_output, label):
    return tf.losses.categorical_crossentropy(label, model_output)


# training
epochs = 10
batch_size = 32
optimizer = tf.optimizers.Adam()

for epoch in range(epochs):
    batch_num = 0
    for batch in batches:
        X, y = tf.cast(batch[0], "float32"), tf.cast(batch[1], "float32")
        with tf.GradientTape() as tape:
            model_out = model(X, weights)
            current_loss = loss(model_out, y)
        grads = tape.gradient(current_loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        batch_num += 1
        if batch_num == 153:
            predictions = tf.argmax(model_out, axis=1)
            train_correct_prediction = tf.equal(predictions, tf.argmax(y, axis=1))
            train_accuracy = tf.reduce_mean(tf.cast(train_correct_prediction, "float"))
            print(f"Currently on epoch {epoch}")
            print(f'The current loss is {np.mean(current_loss)}')
            print(f'The current training accuracy is {train_accuracy}')
print("Finished training \n")


# predict and get accuracy
def evaluate(X, y, weightList):
    X_tf = tf.convert_to_tensor(X)
    y_tf = tf.convert_to_tensor(convert_to_onehot(y))
    softmax_output = model(tf.cast(X_tf, "float32"), weightList)
    output = tf.argmax(softmax_output, axis=1)
    correct_prediction = tf.equal(output, tf.argmax(y_tf, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(f'The accuracy on the dataset is {accuracy}')
    return output, accuracy


evaluate(X_test, Y_test, weights)





