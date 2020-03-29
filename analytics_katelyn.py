import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import statistics as stat

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
    
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    train = pd.read_csv(TRAIN_DATA_PATH)
    
    print("Summary values for all passengers: ")
    print("Most common class:", stat.mode(train["Pclass"]))
    print("Most common sex:", stat.mode(train["Sex"]))
    print("Average age:", np.mean(train["Age"]))
    print("Most common SibSp:", stat.mode(train["SibSp"]))
    print("Most common Parch:", stat.mode(train["Parch"]))
    print("Average fare:", np.mean(train["Fare"]))
    print("Most common embarked:", stat.mode(train["Embarked"]))
    print()
    
    survivors = train.loc[train.Survived == 1]
    print("Summary values for survivors: ")
    print("Most common class:", stat.mode(survivors["Pclass"]))
    print("Most common sex:", stat.mode(survivors["Sex"]))
    print("Average age:", np.mean(survivors["Age"]))
    print("Most common SibSp:", stat.mode(survivors["SibSp"]))
    print("Most common Parch:", stat.mode(survivors["Parch"]))
    print("Average fare:", np.mean(survivors["Fare"]))
    print("Most common embarked:", stat.mode(survivors["Embarked"]))
    print()

    plt.subplot(2, 4, 1)
    
    women = train_data.loc[train_data.Sex == 1]["Survived"]
    rate_women = sum(women)/len(women)
    print("% of women who survived:", rate_women)
    
    men = train_data.loc[train_data.Sex == 0]["Survived"]
    rate_men = sum(men)/len(men)
    print("% of women who survived:", rate_men)
    
    plt.bar(["Female", "Male"], [rate_women, rate_men])
    plt.xlabel('Sex')
    plt.ylabel('Survival Percentage')
    
    
    class1 = train_data.loc[train_data.Pclass == 1]["Survived"]
    rate_class1 = sum(class1)/len(class1)
    print("% of class 1 who survived:", rate_class1)
    
    class2 = train_data.loc[train_data.Pclass == 2]["Survived"]
    rate_class2 = sum(class2)/len(class2)
    print("% of class 2 who survived:", rate_class2)
    
    class3 = train_data.loc[train_data.Pclass == 3]["Survived"]
    rate_class3 = sum(class3)/len(class3)
    print("% of class 3 who survived:", rate_class3)
    
    plt.subplot(2, 4, 2)
    plt.bar(["1st class", "2nd class", "3rd class"], [rate_class1, rate_class2, rate_class3])
    plt.xlabel('Passenger Class')
    
    binSize = 10
    ageBins = []
    rate_ages = []
    #print(len(train_data.loc[train_data.Age == 22]["Survived"]))
    #print(sum(train_data.loc[train_data.Age == 22]["Survived"]))
    prevLen = 0
    prevSum = 0
    for i in range(0,int(80/binSize)):
        ageBin = train_data.loc[train_data.Age <= ((i+1)*binSize)]["Survived"]
        print(sum(ageBin)-prevSum, len(ageBin)-prevLen)
        if(len(ageBin)-prevLen == 0):
            rate_ages.append(0)
        else:
            rate_ages.append((sum(ageBin)-prevSum)/(len(ageBin)-prevLen))
        prevLen = len(ageBin)
        prevSum = sum(ageBin)
        #print(prevLen, " ", prevSum)
    ages = ["10", "20", "30", "40", "50", "60", "70", "80"]
    plt.subplot(2, 4, 3)
    print(rate_ages)
    plt.bar(ages, rate_ages, width = -.9, align = 'edge')
    plt.xlabel('Age range (years)')
    
    
    binSize = 20
    numBins = int(max(train_data['Fare'])/binSize)+1
    fareBins = np.zeros(numBins)
    fareBinsSurvived = np.zeros(numBins)
    for i in range(len(train_data['Fare'])):
        bin = int(train_data['Fare'][i]/binSize)
        fareBins[bin] += 1
        fareBinsSurvived[bin] += train_data['Survived'][i]
     
    #print(fareBinsSurvived/fareBins)
    x = []
    for i in range(numBins):
        x.append((1+i)*binSize)
         
    #x = [0:binSize:numBins*binSize]
    y = fareBinsSurvived/fareBins
    y = y.tolist()
    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = 0
    #print(x)
    
    plt.subplot(2, 4, 4)
    plt.bar(x, y, width=binSize*.9) 
    plt.xlabel('Fare ($)')
    
    
    #print(train_data.loc[train_data.SibSp == 5])
    rate_sibsp = []
    for i in range (0,6):
        rate_sibsp.append(sum(train_data.loc[train_data.SibSp == i]["Survived"])/len(train_data.loc[train_data.SibSp == i]["Survived"]))
    x_sibsp = [0,1,2,3,4,5]
    plt.subplot(2, 4, 5)
    plt.bar(x_sibsp, rate_sibsp) 
    plt.xlabel('# siblings/spouses aboard')
    plt.ylabel('Survival Percentage')
    
    
    rate_parch = []
    for i in range (0,7):
        rate_parch.append(sum(train_data.loc[train_data.Parch == i]["Survived"])/len(train_data.loc[train_data.Parch == i]["Survived"]))
    x_parch = [0,1,2,3,4,5,6]
    plt.subplot(2, 4, 6)
    plt.bar(x_parch, rate_parch) 
    plt.xlabel('# parents/children aboard')
    
    
    rate_embarked = []
    x_embarked = ['C', 'Q', 'S']
    for i in x_embarked:
        print(i)
        rate_embarked.append(sum(train.loc[train.Embarked == i]["Survived"])/len(train.loc[train.Embarked == i]["Survived"]))
    plt.subplot(2, 4, 7)
    plt.bar(x_embarked, rate_embarked) 
    plt.xlabel('Port of departure')
    
    
#     ss1 = train_data.loc[train_data.Pclass == 1]["Survived"]
#     rate_class1 = sum(class1)/len(class1)
#     print("% of class 1 who survived:", rate_class1)
#     
#     class2 = train_data.loc[train_data.Pclass == 2]["Survived"]
#     rate_class2 = sum(class2)/len(class2)
#     print("% of class 2 who survived:", rate_class2)
#     
#     class3 = train_data.loc[train_data.Pclass == 3]["Survived"]
#     rate_class3 = sum(class3)/len(class3)
#     print("% of class 3 who survived:", rate_class3)
#     
    
    
    plt.show()
    #plt.plot()
    
    y = train_data["Survived"]
    
    #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    features = ["Sex", "Age", "Pclass", "Embarked", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    
    print(model.feature_importances_)
    print(model.decision_path(X_test))
    
    from sklearn.tree import export_graphviz
    estimator = model.estimators_[5]
    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot', 
                    feature_names = ["Sex", "Age", "Pclass", "Embarked", "SibSp", "Parch"],
                    class_names = "Survived",
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)
    
    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    #process = subprocess.Popen(command, stdout=tempFile, shell=True)
    #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #set path=%path%;C:\Anaconda3\graphviz-2.38\release\bin
    #dot -Tpng tree.dot -o tree.png
    
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    #score = model.score(X,y)    # test score, since gt not provided for test data
    #print("Accuracy = ", score)
    
# Preprocessing slide (bullets + screenshot) - R
# Features vs. Survival (scatterplot/histogram) - K
# PCA?
# Logistic Regression (confusion matrix, feature weights) - Y
# Random Forest - K
# Neural Net? (confusion matrix, overfitting plot) - R
# Conclusions - Y


