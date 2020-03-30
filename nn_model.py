import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt

# AUXIALIARY FUCNTIONS

# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Opens pickle file and returns stored data object
def open_file(file_name):
	with open(file_name, 'rb') as handle:
		return pickle.load(handle) 


# MODEL CODE

# Returns a trained neural network model that predicts whether a Titanic 
# passenger survived based on a vector of their statistics
def neural_net(Xtr, Ytr, input_shape, alpha, beta1, beta2, B, epochs, best_model_path="best_model.h5"):
    
    d1 = 64
    d2 = 16
    d3 = 4

    callback = tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, mode='min')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(d1, input_shape=input_shape, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(d2, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(d3, activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    optimizer = tf.keras.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(Xtr, Ytr, batch_size=B, epochs=epochs, verbose=2, validation_split=0.1, callbacks=[callback])
    return model, history


# DIMENSIONALITY REDUCTION CODE

def perform_PCA(X, k):
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    sorted_inds = eigvals.argsort()[::-1]
    eigvals = eigvals[sorted_inds]
    eigvecs = eigvecs[:,sorted_inds]
    print(eigvals)
    W = eigvecs[:,0:k]
    X_transf = np.matmul(X,W)
    plt.plot(np.arange(0,len(eigvals)),eigvals)
    plt.show()
    return X_transf, eigvals


# PLOTTING CODE

# Creates a line plot
def create_line_plot(x, ys, title, xlabel, ylabel, line_labels, colors, show_legend=True):
    return


# Creates a heat map
def create_heat_map(X, row_labels, col_labels):
    return


# SUBMISSION CODE

# Creates the output dataframe 
def generate_submission_form(outpath, ids, preds):
    df = pd.DataFrame({'PassengerId':ids, 'Survived':preds})
    df.to_csv(outpath, index=False)
    return


if __name__ == "__main__":
    print("Performing Analysis")
    
    # Data paths
    TRAIN_DATA_PATH = "data/train_data_disc.pickle"
    TRAIN_LABELS_PATH = "data/train_labels.pickle"

    # Train neural network model
    train_data = pd.read_pickle(TRAIN_DATA_PATH)
    train_labels = pd.read_pickle(TRAIN_LABELS_PATH)

    Xtr = train_data.values
    Ytr = train_labels.values
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    n,d = Xtr.shape
    input_shape = (d,)
    B = 32
    epochs = 200
    best_model_path = "best_model.h5"

    model, history = neural_net(Xtr, Ytr, input_shape, alpha, beta1, beta2, B, epochs, best_model_path=best_model_path)
    save_file("history.pickle", history.history)

    # Create predictions
    best_model_path = "best_model.h5"
    TEST_DATA_PATH = "data/test_data_disc.pickle"
    TEST_IDS_PATH = "data/test_ids.pickle"
    test_data = pd.read_pickle(TEST_DATA_PATH)
    test_ids = pd.read_pickle(TEST_IDS_PATH)
    Xte = test_data.values
    model = tf.keras.models.load_model(best_model_path)
    preds = model.predict(Xte)

    # Generate submission file
    outpath = "submission_forms/submissions3.csv"
    generate_submission_form(outpath, test_ids.values, np.argmax(preds, axis=1))