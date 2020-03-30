import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from matplotlib import pyplot as plt


# Saves input python data structure as pickle file in project root
def save_file(file_name, data):
	with open(file_name, 'wb') as handle:
		pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Opens pickle file and returns stored data object
def open_file(file_name):
	with open(file_name, 'rb') as handle:
		return pickle.load(handle) 


def data_analysis(train_data, train_labels, categorical_cols, continuous_cols):
    train_data['Survived'] = train_labels
    survived_df = train_data.loc[train_data['Survived']==1]
    died_df = train_data.loc[train_data['Survived']==0]
    survived_stats = class_statistics(survived_df, categorical_cols, continuous_cols)
    died_stats = class_statistics(died_df, categorical_cols, continuous_cols)
    return survived_stats, died_stats


def class_statistics(class_df, categorical_cols, continuous_cols):
    column_stats = {}
    for col in categorical_cols:
        column_stats[col] = class_df[col].value_counts(normalize=True).to_dict()
    for col in continuous_cols:
        class_col = class_df[col]
        vals = class_col[class_col > 0].values
        column_stats[col] = {"mean": np.mean(vals), "median": np.median(vals), "std": np.std(vals), 
                             "min": np.min(vals), "np.max": np.max(vals), "num_samples": len(vals)}
    return column_stats


def generatePlot(y, x, line_colors, plot_title, x_label, y_label, line_labels):
	fig = plt.figure()
	num_lines = len(y)
	for i in range(0, num_lines):
		plt.plot(x, y[i], color=line_colors[i], label = line_labels[i])
	plt.title(plot_title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend(loc='upper right')
	plt.show()
	return fig


if __name__ == "__main__":
    print("Performing Data Analysis")
    # TRAIN_DATA_PATH = "data/train_data_disc.pickle"
    # TRAIN_LABELS_PATH = "data/train_labels.pickle"
    # train_data = pd.read_pickle(TRAIN_DATA_PATH)
    # train_labels = pd.read_pickle(TRAIN_LABELS_PATH)
    # categorical_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    # continuous_cols = []
    # survived_stats, died_stats = data_analysis(train_data, train_labels, categorical_cols, continuous_cols)
    # print(survived_stats)
    # print(died_stats)

    # history_path = "history1.pickle"
    # history = open_file(history_path)
    # val_loss = history['val_loss']
    # training_loss = history['loss']
    # print(np.argmin(val_loss))
    # x = np.arange(0,len(val_loss))
    # y = [training_loss, val_loss]
    # xlabel = 'Epoch'
    # ylabel = 'Loss'
    # title = 'Neural Network Model Selection'
    # line_labels = ["Training Loss", "Validation Loss"]
    # line_colors = ['b', 'r']

    # generatePlot(y, x, line_colors, title, xlabel, ylabel, line_labels)

    train_data_path = "data/train_data_disc.pickle"
    train_data = pd.read_pickle(train_data_path)
    model_path = 'best_model1.h5'
    model = tf.keras.models.load_model(model_path)
    Xtr = train_data.values
    first_layer_weights = model.layers[0].get_weights()[0]
    # first_layer_biases = model.layers[0].get_weights()[1]
    column_stds = np.std(train_data.values, axis=0)
    # print(train_data.columns.values)
    # print(column_stds)
    # importances = np.sum(first_layer_weights, axis=1)
    # print(importances)

    approx_deriv = first_layer_weights
    for i in range(1, len(model.layers)):
        if i % 2 == 0:
            next_deriv = model.layers[i].get_weights()[0]
            print(next_deriv.shape)
            print(approx_deriv.shape)
            approx_deriv = np.matmul(approx_deriv, next_deriv)

    x = np.arange(8)
    col_names = train_data.columns.values
    importances = np.multiply(np.sum(np.abs(approx_deriv), axis=1), column_stds)
    plt.bar(x, importances)
    plt.xticks(x, col_names)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Neural Network Feature Importances")
    plt.show()

    # session = tf.keras.backend.get_session()
    # gradients = session.run(tf.gradients(model.layers[-1].output, model.input), feed_dict={model.input: Xtr[0,:].reshape((1,8))})
    # print(gradients[0])

