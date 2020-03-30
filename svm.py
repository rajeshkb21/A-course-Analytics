import numpy as np
import pandas as pd
from sklearn import svm
from nn_model import generate_submission_form


if __name__ == "__main__":
    print("Performing Analysis")
    
    # Data paths
    TRAIN_DATA_PATH = "data/train_data_disc.pickle"
    TRAIN_LABELS_PATH = "data/train_labels.pickle"
    TEST_DATA_PATH = "data/test_data_disc.pickle"
    TEST_IDS_PATH = "data/test_ids.pickle"

    train_data = pd.read_pickle(TRAIN_DATA_PATH)
    train_labels = pd.read_pickle(TRAIN_LABELS_PATH)
    test_data = pd.read_pickle(TEST_DATA_PATH)
    test_ids = pd.read_pickle(TEST_IDS_PATH)

    print(test_data.isnull().values.any())

    Xtr = train_data.values
    Xte = test_data.values
    Ytr = train_labels.values
    model = svm.SVC(gamma="auto")
    model.fit(Xtr, Ytr)
    preds = model.predict(Xte)

    outpath = "submission_svm.csv"
    generate_submission_form(outpath, test_ids.values, preds)
