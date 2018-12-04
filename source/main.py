import pandas as pd
from c50.c50 import C50

# Config
summary_txt = '../result/summary.txt'
result_txt = '../result/prediction_result.txt'
result_csv = '../result/prediction_result.csv'
training_data = "../data/train.csv"
test_data = "../data/test.csv"
plot_file = "../result/tree.png"

accTest = False
trials = 1
subset = 0.9  # coefficient of subset size for acc test

# Import data
print("Loading training dataset\n")
dataFrame = pd.read_csv(training_data)
# Replace ? values with null (for rpy2)
dataFrame.replace({"?": None})
# if working on a sample of data
if accTest:
    print('Working on smaller set of data (rest goes for accuracy test)\n')
    dataFrame = dataFrame[:int(len(dataFrame)*subset)]
    acc_set = dataFrame[int(len(dataFrame)*subset):]
    subset = 1  # if acc test so we are working on smaller subset

# Divide dataframe into values and labels


Xtrain = dataFrame.drop(dataFrame.columns[len(dataFrame.columns)-1], axis=1)  # Data
Ytrain = dataFrame.loc[:, dataFrame.columns[len(dataFrame.columns)-1]]  # Labels


# Defining classifier
classifier = C50(Xtrain, Ytrain)

# Training classifier
print(f"Training: Training sample part = {subset}, trials = {trials}\n")
classifier.train(trials=trials, subset=subset)

# Print C50 package summary about decission tree
print(f"Printing summary and saving to file /{summary_txt}\n")
classifier.print_summary(summary_txt)

# Defining test dataframe
print("Loading test set\n")
X = pd.read_csv(test_data)

# Save result of prediction on pd.read_csv('../data/test.csv') dataframe (just predicted values)
print(f"Saving predicted values in {result_txt}\n")
classifier.predict_and_save(X, result_txt)

# Save result of prediction with predicting rows on csvread_csv('../data/test.csv') dataframe
print(f"Saving predicted values with predicting rows in {result_csv}\n")
classifier.predict_to_csv(X, result_csv)


# Accuracy test
if accTest:
    Xtest = acc_set.drop(acc_set.columns[len(acc_set.columns)-1], axis=1)  # Data
    Xtest = Xtest.reset_index(drop=True)
    Ytest = acc_set.loc[:, acc_set.columns[len(acc_set.columns)-1]].values  # Labels
    Y = classifier.predict(Xtest)
    count = 0
    for i in range(0, len(Y)):
        if Y[i] == str(Ytest[i]):
            count += 1
    print(f"Good/bad {count/(len(Y))} coefficient")