import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, log_loss

########################################################################################################################
#                               1. TRAIN AND TEST DATA LOADING AND PREPROCESSING                                       #
########################################################################################################################

# Load the train and test sets and make dataframe out of them
train_csv = 'loan_train.csv'
test_csv = 'loan_test.csv'
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Convert the columns with date and time into a 'datetime' format
df_train['due_date'] = pd.to_datetime(df_train['due_date'])
df_test['due_date'] = pd.to_datetime(df_test['due_date'])
df_train['effective_date'] = pd.to_datetime(df_train['effective_date'])
df_test['effective_date'] = pd.to_datetime(df_test['effective_date'])

# Convert the 'effective date' column into the day of week
df_train['dayofweek'] = df_train['effective_date'].dt.dayofweek
df_test['dayofweek'] = df_test['effective_date'].dt.dayofweek

# Convert the weekdays into a numeric binary
df_train['weekend'] = df_train['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
df_test['weekend'] = df_test['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)

# Convert the genders into a numeric binary
df_train['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
df_test['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

# Define the normalized feature set
feature_train = df_train[['Principal', 'terms', 'age', 'Gender', 'weekend']]
feature_test = df_test[['Principal', 'terms', 'age', 'Gender', 'weekend']]
feature_train = pd.concat([feature_train, pd.get_dummies(df_train['education'])], axis=1)
feature_test = pd.concat([feature_test, pd.get_dummies(df_test['education'])], axis=1)
feature_train.drop(['Master or Above'], axis=1, inplace=True)
feature_test.drop(['Master or Above'], axis=1, inplace=True)
X_train = StandardScaler().fit(feature_train).transform(feature_train)
X_test = StandardScaler().fit(feature_test).transform(feature_test)

# Define the actual labels set
y_train = df_train['loan_status'].values
y_test = df_test['loan_status'].values

########################################################################################################################
#                                    2. KNN (K-NEAREST NEIGHBOR) CLASSIFICATION                                        #
########################################################################################################################

# Iterate over the number of neighbors in order to find the optimal number
n_comp_X_train, n_comp_X_test, n_comp_y_train, n_comp_y_test = train_test_split(X_train, y_train, test_size=0.2,
                                                                                random_state=4)
n_max = 10
mean_accuracy = np.zeros((n_max-1))
for n in range(1, n_max):

    # Create the model for n neighbours
    knn_model_n = KNeighborsClassifier(n_neighbors=n)
    knn_model_n.fit(n_comp_X_train, n_comp_y_train)

    # Calculate the predicted value for n neighbours
    knn_y_hat_n = knn_model_n.predict(n_comp_X_test)

    # Evaluate the accuracy for n neighbors
    mean_accuracy[n-1] = accuracy_score(n_comp_y_test, knn_y_hat_n)

# Select the number of neighbors for which the mean accuracy is maximum
n_optimal = mean_accuracy.argmax() + 1
print(f'The optimal number of nearest neighbors to be considered is {n_optimal}')

# Create the model for the optimal number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=n_optimal)
knn_model.fit(X_train, y_train)

# Calculate the predicted labels set for the test feature set
knn_y_hat = knn_model.predict(X_test)

# Evaluate the prediction according to Jaccard and F1 scores
knn_jac = jaccard_score(y_test, knn_y_hat, pos_label='PAIDOFF')
knn_f1 = f1_score(y_test, knn_y_hat, average='weighted')

########################################################################################################################
#                                         3. DECISION-TREE CLASSIFICATION                                              #
########################################################################################################################

# Create the model and fit it with data
tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=7)
tree_model.fit(X_train, y_train)

# Calculate the predicted labels set for the test feature set
tree_y_hat = tree_model.predict(X_test)

# Evaluate the prediction according to Jaccard and F1 scores
tree_jac = jaccard_score(y_test, tree_y_hat, pos_label='PAIDOFF')
tree_f1 = f1_score(y_test, tree_y_hat, average='weighted')

########################################################################################################################
#                                   4. SVM (SUPPORT VECTOR MACHINE) CLASSIFICATION                                     #
########################################################################################################################

# Create the model and fit it with data
svm_model = SVC(kernel='rbf')  # we use Radial Basis Function (RBF) as kernel
svm_model.fit(X_train, y_train)

# Calculate the predicted labels set for the test feature set
svm_y_hat = svm_model.predict(X_test)

# Evaluate the prediction according to Jaccard and F1 scores
svm_jac = jaccard_score(y_test, svm_y_hat, pos_label='PAIDOFF')
svm_f1 = f1_score(y_test, svm_y_hat, average='weighted')

########################################################################################################################
#                                       5. LOGISTIC REGRESSION CLASSIFICATION                                          #
########################################################################################################################

# Create the model and fit it with data
log_reg_model = LogisticRegression(C=0.01, solver='liblinear')
log_reg_model.fit(X_train, y_train)

# Calculate the predicted labels set for the test feature set
log_reg_y_hat = log_reg_model.predict(X_test)
log_reg_y_hat_prob = log_reg_model.predict_proba(X_test)

# Evaluate the prediction according to Jaccard and F1 scores and logarithmic regression
log_reg_jac = jaccard_score(y_test, log_reg_y_hat, pos_label='PAIDOFF')
log_reg_f1 = f1_score(y_test, log_reg_y_hat, average='weighted')
log_reg_ll = log_loss(y_test, log_reg_y_hat_prob)

########################################################################################################################
#                                                6. DATA REPORTING                                                     #
########################################################################################################################

# Store the data to be reported in a dictionary
dict_report = {'Algorithm':  ['KNN', 'Decision Tree', 'SVM', 'LogisticRegression'],
               'Jaccard-score': [knn_jac, tree_jac, svm_jac, log_reg_jac],
               'F1-score': [knn_f1, tree_f1, svm_f1, log_reg_f1],
               'LogLoss': ['NA', 'NA', 'NA', log_reg_ll]}

# Create a report dataframe out of the dictionary and print it out
df_report = pd.DataFrame(dict_report, columns=['Algorithm', 'Jaccard-score', 'F1-score', 'LogLoss'])
df_report.set_index('Algorithm', inplace=True)
print(df_report)
