import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
# from sklearn.metrics import r2_score
# from sklearn.metrics import plot_confusion_matrix

# Use pandas functions to preprocess the data
def preprocess_data(train_data_path, test_data_path):
    
    # extract data
    train_data = pd.read_csv(train_data_path)
    train_labels = train_data['Survived'].values
    test_data = pd.read_csv(test_data_path)

    # If data is missing replace with unknown flag -1 or 'UNK'
    train_data = train_data.fillna(-1)
    test_data = test_data.fillna(-1)
    train_data['Cabin'] = train_data['Cabin'].replace(-1, 'UNK')
    test_data['Cabin'] = test_data['Cabin'].replace(-1, 'UNK')

    # define mappings of categorical features to numerical values
    mf_mapping = {-1: -1, 'male': 0, 'female': 1}
    embarked_mapping = {-1:-1, 'S': 0, 'C': 1, 'Q': 2}
    cabin_mapping = create_cabin_mapping(train_data['Cabin'])

    # apply the mappings to their respective columns
    train_data['Sex'] = train_data['Sex'].map(mf_mapping)
    test_data['Sex'] = test_data['Sex'].map(mf_mapping)
    train_data['Cabin'] = train_data['Cabin'].apply(cabin_mapping_function, args=(cabin_mapping,))
    test_data['Cabin'] = test_data['Cabin'].apply(cabin_mapping_function, args=(cabin_mapping,))
    train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
    test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)
    train_data['Ticket'] = train_data['Ticket'].apply(ticket_mapping_function)
    test_data['Ticket'] = test_data['Ticket'].apply(ticket_mapping_function)
    
    # drop uninformative columns
    train_data.drop(columns=['PassengerId', 'Name'])
    test_data.drop(columns=['PassengerId', 'Name'])
    
    # Save dataframes as pickle files
    train_data.to_pickle("train_data.pickle")
    test_data.to_pickle("test_data.pickle")
    data = {'train': train_data, 'test': test_data}
    return data


# If the ticket contains a prefix, assign a 1 else assign 0
def ticket_mapping_function(val):
    split_val = val.split(' ')
    if len(split_val) > 1:
        return 1
    else:
        return 0


# Create a dictionary that captures the first letter of the cabin number
def create_cabin_mapping(cabin_col):
    unique_vals = np.unique(cabin_col.values).tolist()
    cabin_mapping = {'U': 0}
    i = 1
    for val in unique_vals:
        if type(val) == str:
            cabin_char = val[0]
            if cabin_char not in cabin_mapping:
                cabin_mapping[cabin_char] = i
                i += 1
    return cabin_mapping


# Function to maps string cabin numbers to the cabin id:
def cabin_mapping_function(val, cabin_mapping):
    if type(val) == str:
        cabin_char = val[0]
        if cabin_char in cabin_mapping:
            return cabin_mapping[cabin_char]
    else:
        return -1


if __name__ == '__main__':
    TRAIN_DATA_PATH = 'train.csv'
    TEST_DATA_PATH = 'test.csv'
    data = preprocess_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    train_data = data['train']
    test_data = data['test']
    
    y = train_data["Survived"]
    
    # features = ["Pclass", "Sex", "Age","Embarked", "Fare","SibSp", "Parch"]
    features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
    X = train_data[features]
    X_test = test_data[features]
    model = LogisticRegression(random_state=1, solver='liblinear').fit(X,y)
    predictions = model.predict(X_test)
    train_predictions = model.predict(X)
    
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission_yohan.csv', index=False)
    score = model.score(X,y)
    print("Accuracy =", score)
    # r_sq = sklearn.metrics.r2_score(y,train_predictions)
    # print("R2 = ", r_sq)

    # Plot normalized confusion matrix
    title = "Training Data Confusion Matrix"
    disp = sklearn.metrics.plot_confusion_matrix(model, X, y,
                                 display_labels=["Survived","Dead"],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title(title)
    # print(title)
    # print(disp.confusion_matrix)
    plt.show()

    # Get model parameters
    coeffs = model.coef_
    print("Coefficients:",coeffs)