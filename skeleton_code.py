import numpy as np
import pandas as pd
import pickle
import sklearn
import tensorflow as tf 
from matplotlib import pyplot as plt


# Here is a neural network model for processing images of handwritten digits
# A neural network is a ML model that takes an input and processes it through a series of layers
# Each layer takes a vector input and performs a matrix multiplication. A nonlinearity is applied
#   to each element of that vector. The resulting vector is then passed to the next layer.
# This forward propagation process continues until the output is generated 
def example_neural_network(input_shape):

    # model layer structure
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(d1, activation=tf.nn.relu),
        tf.keras.layers.Dense(d1, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # define model optimization function
    # Here, we are training the model using a gradient descent algorithm named Adam
    # Hyperparameters: alpha is the learning rate
    #                  beta_1 is the first moment coefficient (set to 0.99)
    #                  beta_2 is the second moment coefficient (set to 0.999)
    # Our loss function is categorical crossentropy loss, which is used to train multiclass models
    optimizer = tf.keras.optimizers.Adam(lr=alpha, beta_1=rho1, beta_2=rho2)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train the model
    # we are using minibatching with batch size B
    # we are splitting the training data into a held-out validation set along a 90-10 basis
    history = model.fit(Xs, Ys, batch_size=B, epochs=epochs, verbose=2, validation_split=0.1)
    return model, history


#TODO
def actual_NN_model():
    return


#TODO 
# Use pandas functions to preprocess the data
def preprocess_data(datapath):
    data = pd.csvread(datapath)
    return


# PCA is a dimensionality reduction technique. 
# Given input parameter 'k', PCA finds the top k directions of the feature space on which the data 
#   exhibits the greatest variance.
# PCA then projects the data onto these top 'k' directions
# This creates a lower dimensional representation of the data that preserves as much information 
#   as possible
# Choice of 'k' is important and there is a cool way to select this
def perform_PCA(X, k):
    cov = np.cov(X)
    eigvecs, eigvals = np.linalg.eigvals(cov)
    sorted_inds = eigvals.argsort()[::-1]
    eigvals = eigvals[sorted_inds]
    eigvecs = eigvals[:,sorted_inds]
    W = eigvecs[:,0:k]
    X_transf = np.matmul(X,W)
    return X_transf, eigvals


# TODO
# For all other models, take a look at the sklearn docs


# TODO
def generate_submission_form():
    return