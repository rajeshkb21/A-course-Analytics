import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
    
    
    women = train_data.loc[train_data.Sex == 1]["Survived"]
    rate_women = sum(women)/len(women)
    print("% of women who survived:", rate_women)
    
    men = train_data.loc[train_data.Sex == 0]["Survived"]
    rate_men = sum(men)/len(men)
    print("% of women who survived:", rate_men)
    
    class1 = train_data.loc[train_data.Pclass == 1]["Survived"]
    rate_class1 = sum(class1)/len(class1)
    print("% of class 1 who survived:", rate_class1)
    
    class2 = train_data.loc[train_data.Pclass == 2]["Survived"]
    rate_class2 = sum(class2)/len(class2)
    print("% of class 2 who survived:", rate_class2)
    
    class3 = train_data.loc[train_data.Pclass == 3]["Survived"]
    rate_class3 = sum(class3)/len(class3)
    print("% of class 3 who survived:", rate_class3)
    
#     binSize = 20
#     numBins = int(max(train_data['Fare'])/binSize)+1
#     fareBins = np.zeros(numBins)
#     fareBinsSurvived = np.zeros(numBins)
#     for i in range(len(train_data['Fare'])):
#         bin = int(train_data['Fare'][i]/binSize)
#         fareBins[bin] += 1
#         fareBinsSurvived[bin] += train_data['Survived'][i]
#     
#     #print(fareBinsSurvived/fareBins)
#     x = []
#     for i in range(numBins):
#         x.append((1+i)*binSize)
#         
#     #x = [0:binSize:numBins*binSize]
#     y = fareBinsSurvived/fareBins
#     y = y.tolist()
#     for i in range(len(y)):
#         if np.isnan(y[i]):
#             y[i] = 0
#     print(x)
#     
#     plt.bar(x, y, width=binSize*.9) 
#     plt.xlabel('Fare ($)')
#     plt.ylabel('Survival Percentage')
#     #plt.show()
#     #plt.plot()
    
    y = train_data["Survived"]
    
    features = ["Sex", "Fare", "Age"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    #print(test_data)
    #y_test = test_data["Survived"]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    print(model.feature_importances_)
    
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    #score = model.score(X,y)    # test score, since gt not provided for test data
    #print("Accuracy = ", score)
    



