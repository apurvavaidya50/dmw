import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("C:\Users\user\Desktop\Dmw_proj\\test.py")
print(dataset.head())
print(dataset.shape)
column_normalize_selector = ['Age', 'education_num', 'capital_gain','capital_loss','hours_per_week']
scalar = MinMaxScaler()
dataset[column_normalize_selector] = scalar.fit_transform(dataset[column_normalize_selector])
le = LabelEncoder()
dataset['native_country'] = le.fit_transform(dataset['native_country'])
dataset['race'] = le.fit_transform(dataset['race'])
dataset['sex'] = le.fit_transform(dataset['sex'])
dataset['Workclass'] = le.fit_transform(dataset['Workclass'])
dataset['education_level'] = le.fit_transform(dataset['education_level'])
dataset['Marital_status'] = le.fit_transform(dataset['Marital_status'])
dataset['occupation'] = le.fit_transform(dataset['occupation'])
dataset['relationship'] = le.fit_transform(dataset['relationship'])
dataset['income'] = le.fit_transform(dataset['income'])

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model_DecisionTree = DecisionTreeClassifier()
model_DecisionTree.fit(X_train, y_train)
print(model_DecisionTree.score(X_test, y_test))
print("Confusion matrix for decision tree classifier: ")
print(confusion_matrix(y_test, model_DecisionTree.predict(X_test)))

# Run Naive Bayes model
model_Bayes = GaussianNB()
model_Bayes.fit(X_train, y_train)
print(model_Bayes.score(X_test, y_test))
print("Confusion matrix for Naive Bayes classifier:")
print(confusion_matrix(y_test, model_Bayes.predict(X_test)))

# Run Logistic Regression model
model_LogisticRegression = LogisticRegression()
model_LogisticRegression.fit(X_train, y_train)
print(model_LogisticRegression.score(X_test, y_test))
print("Confusion matrix for Logistic Regression classifier")
print(confusion_matrix(y_test, model_LogisticRegression.predict(X_test)))

# Run support vector machine
model_LinearSVC = LinearSVC()
model_LinearSVC.fit(X_train, y_train)
print(model_LinearSVC.score(X_test, y_test))
print("Confusion matrix for SVC : ")
print(confusion_matrix(y_test, model_LinearSVC.predict(X_test)))

# draw accuracy plot

accuracy_list = [model_DecisionTree.score(X_test, y_test), model_Bayes.score(X_test, y_test),
                 model_LogisticRegression.score(X_test, y_test), model_LinearSVC.score(X_test, y_test)]
label = ['Decision Tree', "NaiveBayes", "LogisticRegression", "LinearSVC"]
plt.bar(np.arange(len(label)), accuracy_list)
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.xticks(np.arange(len(label)), label)
plt.show()
