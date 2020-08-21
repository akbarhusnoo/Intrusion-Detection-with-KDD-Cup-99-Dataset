# -*- coding: utf-8 -*-
"""
Spyder Editor

Student Name: Muhammad Akbar Husnoo
Student ID: 219306444
Task Name: 5.1D/HD: End-to-End Project Delivery on Cyber-security Data Analytics
"""

import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


dataset_root = '/Users/akbar/Downloads/Demo_code/NSL-KDD-Dataset/NSL-KDD-Dataset'

train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')


# Original KDD dataset feature names obtained from 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
len(header_names)
print(header_names)
# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names 
# file, but it is supposed to be a binary feature according to the 
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
# file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types



category = defaultdict(list)
category['benign'].append('normal')

with open('/Users/akbar/Downloads/Demo_code/NSL-KDD-Dataset/NSL-KDD-Dataset/training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v,k) for k in category for v in category[k])


train_df = pd.read_csv(train_file, names=header_names)
print(len(train_df))
train_df['attack_category'] = train_df['attack_type'] \
                                .map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)
    
test_df = pd.read_csv(test_file, names=header_names)
print(len(test_df))
test_df['attack_category'] = test_df['attack_type'] \
                                .map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)


train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()
print(test_attack_cats)

train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)

plt.xlabel('NUMBER OF SAMPLES')
plt.ylabel('ATTACK CATEGORIES')
train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=20, title = 'BARCHART SHOWING THE NUMBER OF SAMPLES FOR EACH ATTACK CATEGORY IN KDD TRAIN DATASET')

test_attack_types.plot(kind='barh', figsize=(20,10), fontsize=15)

test_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)

# Let's take a look at the binary features
# By definition, all of these features should have a min of 0.0 and a max of 1.0
#execute the commands in console

train_df[binary_cols].describe().transpose()


# Wait a minute... the su_attempted column has a max value of 2.0?

train_df.groupby(['su_attempted']).size()

# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.groupby(['su_attempted']).size()


# Next, we notice that the num_outbound_cmds column only takes on one value!

train_df.groupby(['num_outbound_cmds']).size()

# Now, that's not a very useful feature - let's drop it from the dataset

train_df.drop('num_outbound_cmds', axis = 1, inplace=True)
test_df.drop('num_outbound_cmds', axis = 1, inplace=True)
numeric_cols.remove('num_outbound_cmds')


"""
Data Preparation

"""
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)




combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))

#execute the commands in console
train_x.describe()
train_x['duration'].describe()
# Experimenting with StandardScaler on the single 'duration' feature
from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()

# Experimenting with MinMaxScaler on the single 'duration' feature
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
pd.Series(min_max_scaled_durations.flatten()).describe()

# Experimenting with RobustScaler on the single 'duration' feature
from sklearn.preprocessing import RobustScaler

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
pd.Series(robust_scaled_durations.flatten()).describe()

# Let's proceed with StandardScaler- Apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])
    
train_x.describe()


train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)


"""
Step 6: Predictive Modelling

"""
import datetime as dt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, zero_one_loss, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
import seaborn as sn
import matplotlib.pyplot as plt

### Implementation of Gaussian Naive Bayes (Accuracy Achieved = 0.7835787792760823) ###
from sklearn.naive_bayes import GaussianNB
#include smoothing
gaussian_naives_bayes_classifier = GaussianNB(var_smoothing = 0.0992) 
#get model building start time
gaussian_naives_bayes_start_time = dt.datetime.now()
#fit data to classifier
gaussian_naives_bayes_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_gaussian_naives_bayes = gaussian_naives_bayes_classifier.predict(test_x)
#get model building end time
gaussian_naives_bayes_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_gaussian_naives_bayes = confusion_matrix(test_Y, predict_y_using_gaussian_naives_bayes)
error = zero_one_loss(test_Y, predict_y_using_gaussian_naives_bayes)

print('=' * 33)
print('|  GAUSSIAN NAIVES BAYES MODEL  |')
print('=' * 33)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_gaussian_naives_bayes) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_gaussian_naives_bayes_df = pd.DataFrame(confusion_matrix_from_gaussian_naives_bayes, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for Gaussian Naives Bayes Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_gaussian_naives_bayes_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_gaussian_naive_bayes = confusion_matrix_from_gaussian_naives_bayes.astype('float') / confusion_matrix_from_gaussian_naives_bayes.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_gaussian_naive_bayes.diagonal()))

#Compute False Positive Rate per label
tp_gaussian_naives_bayes = np.diag(confusion_matrix_from_gaussian_naives_bayes)
print('True Positive Per Label: ' + str(tp_gaussian_naives_bayes))

fp_gaussian_naives_bayes = np.sum(confusion_matrix_from_gaussian_naives_bayes, axis=0) - tp_gaussian_naives_bayes
print('False Positive Per Label: ' + str(fp_gaussian_naives_bayes))

fn_gaussian_naives_bayes = np.sum(confusion_matrix_from_gaussian_naives_bayes, axis=1) - tp_gaussian_naives_bayes
print('False Negative Per Label: ' + str(fn_gaussian_naives_bayes))

benign_tn = np.sum(confusion_matrix_from_gaussian_naives_bayes) - (tp_gaussian_naives_bayes[0]+ fp_gaussian_naives_bayes[0] + fn_gaussian_naives_bayes[0])
dos_tn = np.sum(confusion_matrix_from_gaussian_naives_bayes) - (tp_gaussian_naives_bayes[1]+ fp_gaussian_naives_bayes[1] + fn_gaussian_naives_bayes[1])
probe_tn = np.sum(confusion_matrix_from_gaussian_naives_bayes) - (tp_gaussian_naives_bayes[2]+ fp_gaussian_naives_bayes[2] + fn_gaussian_naives_bayes[2])
r2l_tn = np.sum(confusion_matrix_from_gaussian_naives_bayes) - (tp_gaussian_naives_bayes[3]+ fp_gaussian_naives_bayes[3] + fn_gaussian_naives_bayes[3])
u2r_tn = np.sum(confusion_matrix_from_gaussian_naives_bayes) - (tp_gaussian_naives_bayes[4]+ fp_gaussian_naives_bayes[4] + fn_gaussian_naives_bayes[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_gaussian_naives_bayes[0]/ (fp_gaussian_naives_bayes[0] + benign_tn)
dos_fpr = fp_gaussian_naives_bayes[1]/ (fp_gaussian_naives_bayes[1] + dos_tn)
probe_fpr = fp_gaussian_naives_bayes[2]/ (fp_gaussian_naives_bayes[2] + probe_tn)
r2l_fpr = fp_gaussian_naives_bayes[3]/ (fp_gaussian_naives_bayes[3] + r2l_tn)
u2r_fpr = fp_gaussian_naives_bayes[4]/ (fp_gaussian_naives_bayes[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_gaussian_naives_bayes = classification_report(test_Y, predict_y_using_gaussian_naives_bayes)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_gaussian_naives_bayes) + '\n')

#compute accuracy metric
accuracy_gaussian_naives_bayes = accuracy_score(test_Y, predict_y_using_gaussian_naives_bayes)
print('Accuracy Score = ' + str(accuracy_gaussian_naives_bayes) + '\n')

#compute precision metric
precision_gaussian_naives_bayes = precision_score(test_Y, predict_y_using_gaussian_naives_bayes, average = 'weighted')
print('Precision Score = ' + str(precision_gaussian_naives_bayes) + '\n')

#compute recall metric
recall_gaussian_naives_bayes = recall_score(test_Y, predict_y_using_gaussian_naives_bayes, average = 'weighted')
print('Recall Score = ' + str(recall_gaussian_naives_bayes) + '\n')

#compute F1-score metric
f1_score_gaussian_naives_bayes = f1_score(test_Y, predict_y_using_gaussian_naives_bayes, average = 'weighted')
print('F1 Score = ' + str(f1_score_gaussian_naives_bayes) + '\n')

#compute Total Model built time
gaussian_naives_bayes_built_time = gaussian_naives_bayes_end_time - gaussian_naives_bayes_start_time
print('Model Built Time = ' + str(gaussian_naives_bayes_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_gaussian_naive_bayes = np.sum(fp_gaussian_naives_bayes)/ (np.sum(fp_gaussian_naives_bayes) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_gaussian_naive_bayes) + '\n')


### Implementation of Logistic Regression (Accuracy Achieved = 0.7530606813342796) ###

from sklearn.linear_model import LogisticRegression
logistic_regression_classifier = LogisticRegression(solver = 'lbfgs')
#get model building start time
logistic_regression_start_time = dt.datetime.now()
#fit data to classifier
logistic_regression_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_logistic_regression = logistic_regression_classifier.predict(test_x)
#get model building end time
logistic_regression_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_logistic_regression = confusion_matrix(test_Y, predict_y_using_logistic_regression)
error = zero_one_loss(test_Y, predict_y_using_logistic_regression)

print('=' * 31)
print('|  LOGISTIC REGRESSION MODEL  |')
print('=' * 31)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_logistic_regression) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_logistic_regression_df = pd.DataFrame(confusion_matrix_from_logistic_regression, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for Logistic Regression Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_logistic_regression_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_logistic_regression = confusion_matrix_from_logistic_regression.astype('float') / confusion_matrix_from_logistic_regression.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_logistic_regression.diagonal()))

#Compute False Positive Rate per label
tp_logistic_regression = np.diag(confusion_matrix_from_logistic_regression)
print('True Positive Per Label: ' + str(tp_logistic_regression))

fp_logistic_regression = np.sum(confusion_matrix_from_logistic_regression, axis=0) - tp_logistic_regression
print('False Positive Per Label: ' + str(fp_logistic_regression))

fn_logistic_regression = np.sum(confusion_matrix_from_logistic_regression, axis=1) - tp_logistic_regression
print('False Negative Per Label: ' + str(fn_logistic_regression))

benign_tn = np.sum(confusion_matrix_from_logistic_regression) - (tp_logistic_regression[0]+ fp_logistic_regression[0] + fn_logistic_regression[0])
dos_tn = np.sum(confusion_matrix_from_logistic_regression) - (tp_logistic_regression[1]+ fp_logistic_regression[1] + fn_logistic_regression[1])
probe_tn = np.sum(confusion_matrix_from_logistic_regression) - (tp_logistic_regression[2]+ fp_logistic_regression[2] + fn_logistic_regression[2])
r2l_tn = np.sum(confusion_matrix_from_logistic_regression) - (tp_logistic_regression[3]+ fp_logistic_regression[3] + fn_logistic_regression[3])
u2r_tn = np.sum(confusion_matrix_from_logistic_regression) - (tp_logistic_regression[4]+ fp_logistic_regression[4] + fn_logistic_regression[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_logistic_regression[0]/ (fp_logistic_regression[0] + benign_tn)
dos_fpr = fp_logistic_regression[1]/ (fp_logistic_regression[1] + dos_tn)
probe_fpr = fp_logistic_regression[2]/ (fp_logistic_regression[2] + probe_tn)
r2l_fpr = fp_logistic_regression[3]/ (fp_logistic_regression[3] + r2l_tn)
u2r_fpr = fp_logistic_regression[4]/ (fp_logistic_regression[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_logistic_regression = classification_report(test_Y, predict_y_using_logistic_regression)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_logistic_regression) + '\n')

#compute accuracy metric
accuracy_logistic_regression = accuracy_score(test_Y, predict_y_using_logistic_regression)
print('Accuracy Score = ' + str(accuracy_logistic_regression) + '\n')

#compute precision metric
precision_logistic_regression = precision_score(test_Y, predict_y_using_logistic_regression, average = 'weighted')
print('Precision Score = ' + str(precision_logistic_regression) + '\n')

#compute recall metric
recall_logistic_regression = recall_score(test_Y, predict_y_using_logistic_regression, average = 'weighted')
print('Recall Score = ' + str(recall_logistic_regression) + '\n')

#compute F1-score metric
f1_score_logistic_regression = f1_score(test_Y, predict_y_using_logistic_regression, average = 'weighted')
print('F1 Score = ' + str(f1_score_logistic_regression) + '\n')

#compute Total Model built time
logistic_regression_built_time = logistic_regression_end_time - logistic_regression_start_time
print('Model Built Time = ' + str(logistic_regression_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_logistic_regression = np.sum(fp_logistic_regression)/ (np.sum(fp_logistic_regression) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_logistic_regression) + '\n')



### Implementation of Support Vector Machine Classifier (Accuracy Achieved = 0.7786550745209369) ###

from sklearn.svm import SVC
support_vector_classifier = SVC(probability=True, C=10.0, gamma=0.001, kernel='rbf')
#get model building start time
support_vector_start_time = dt.datetime.now()
#fit data to classifier
support_vector_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_support_vector = support_vector_classifier.predict(test_x)
#get model building end time
support_vector_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_support_vector = confusion_matrix(test_Y, predict_y_using_support_vector)
error = zero_one_loss(test_Y, predict_y_using_support_vector)

print('=' * 34)
print('|  SUPPORT VECTOR MACHINE MODEL  |')
print('=' * 34)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_support_vector) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_support_vector_df = pd.DataFrame(confusion_matrix_from_support_vector, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for Support Vector Machine Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_support_vector_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_support_vector = confusion_matrix_from_support_vector.astype('float') / confusion_matrix_from_support_vector.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_support_vector.diagonal()))

#Compute False Positive Rate per label
tp_support_vector = np.diag(confusion_matrix_from_support_vector)
print('True Positive Per Label: ' + str(tp_support_vector))

fp_support_vector = np.sum(confusion_matrix_from_support_vector, axis=0) - tp_support_vector
print('False Positive Per Label: ' + str(fp_support_vector))

fn_support_vector = np.sum(confusion_matrix_from_support_vector, axis=1) - tp_support_vector
print('False Negative Per Label: ' + str(fn_support_vector))

benign_tn = np.sum(confusion_matrix_from_support_vector) - (tp_support_vector[0]+ fp_support_vector[0] + fn_support_vector[0])
dos_tn = np.sum(confusion_matrix_from_support_vector) - (tp_support_vector[1]+ fp_support_vector[1] + fn_support_vector[1])
probe_tn = np.sum(confusion_matrix_from_support_vector) - (tp_support_vector[2]+ fp_support_vector[2] + fn_support_vector[2])
r2l_tn = np.sum(confusion_matrix_from_support_vector) - (tp_support_vector[3]+ fp_support_vector[3] + fn_support_vector[3])
u2r_tn = np.sum(confusion_matrix_from_support_vector) - (tp_support_vector[4]+ fp_support_vector[4] + fn_support_vector[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_support_vector[0]/ (fp_support_vector[0] + benign_tn)
dos_fpr = fp_support_vector[1]/ (fp_support_vector[1] + dos_tn)
probe_fpr = fp_support_vector[2]/ (fp_support_vector[2] + probe_tn)
r2l_fpr = fp_support_vector[3]/ (fp_support_vector[3] + r2l_tn)
u2r_fpr = fp_support_vector[4]/ (fp_support_vector[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_support_vector = classification_report(test_Y, predict_y_using_support_vector)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_support_vector) + '\n')

#compute accuracy metric
accuracy_support_vector = accuracy_score(test_Y, predict_y_using_support_vector)
print('Accuracy Score = ' + str(accuracy_support_vector) + '\n')

#compute precision metric
precision_support_vector = precision_score(test_Y, predict_y_using_support_vector, average = 'weighted')
print('Precision Score = ' + str(precision_support_vector) + '\n')

#compute recall metric
recall_support_vector = recall_score(test_Y, predict_y_using_support_vector, average = 'weighted')
print('Recall Score = ' + str(recall_support_vector) + '\n')

#compute F1-score metric
f1_score_support_vector = f1_score(test_Y, predict_y_using_support_vector, average = 'weighted')
print('F1 Score = ' + str(f1_score_support_vector) + '\n')

#compute Total Model built time
support_vector_built_time = support_vector_end_time - support_vector_start_time
print('Model Built Time = ' + str(support_vector_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_support_vector = np.sum(fp_support_vector)/ (np.sum(fp_support_vector) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_support_vector) + '\n')



### Implementation of AdaBoost Classifier (Accuracy Achieved = 0.7458303761533002) ###

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
AdaBoost_classifier = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 100, random_state = 5) 
#get model building start time
AdaBoost_start_time = dt.datetime.now()
#fit data to classifier
AdaBoost_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_AdaBoost = AdaBoost_classifier.predict(test_x)
#get model building end time
AdaBoost_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_AdaBoost = confusion_matrix(test_Y, predict_y_using_AdaBoost)
error = zero_one_loss(test_Y, predict_y_using_AdaBoost)

##
accuracy_AdaBoost = accuracy_score(test_Y, predict_y_using_AdaBoost)
print('Accuracy Score = ' + str(accuracy_AdaBoost) + '\n')
##
print('=' * 20)
print('|  ADABOOST MODEL  |')
print('=' * 20)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_AdaBoost) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_AdaBoost_df = pd.DataFrame(confusion_matrix_from_AdaBoost, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for AdaBoost Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_AdaBoost_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_AdaBoost = confusion_matrix_from_AdaBoost.astype('float') / confusion_matrix_from_AdaBoost.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_AdaBoost.diagonal()))

#Compute False Positive Rate per label
tp_AdaBoost = np.diag(confusion_matrix_from_AdaBoost)
print('True Positive Per Label: ' + str(tp_AdaBoost))

fp_AdaBoost = np.sum(confusion_matrix_from_AdaBoost, axis=0) - tp_AdaBoost
print('False Positive Per Label: ' + str(fp_AdaBoost))

fn_AdaBoost = np.sum(confusion_matrix_from_AdaBoost, axis=1) - tp_AdaBoost
print('False Negative Per Label: ' + str(fn_AdaBoost))

benign_tn = np.sum(confusion_matrix_from_AdaBoost) - (tp_AdaBoost[0]+ fp_AdaBoost[0] + fn_AdaBoost[0])
dos_tn = np.sum(confusion_matrix_from_AdaBoost) - (tp_AdaBoost[1]+ fp_AdaBoost[1] + fn_AdaBoost[1])
probe_tn = np.sum(confusion_matrix_from_AdaBoost) - (tp_AdaBoost[2]+ fp_AdaBoost[2] + fn_AdaBoost[2])
r2l_tn = np.sum(confusion_matrix_from_AdaBoost) - (tp_AdaBoost[3]+ fp_AdaBoost[3] + fn_AdaBoost[3])
u2r_tn = np.sum(confusion_matrix_from_AdaBoost) - (tp_AdaBoost[4]+ fp_AdaBoost[4] + fn_AdaBoost[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_AdaBoost[0]/ (fp_AdaBoost[0] + benign_tn)
dos_fpr = fp_AdaBoost[1]/ (fp_AdaBoost[1] + dos_tn)
probe_fpr = fp_AdaBoost[2]/ (fp_AdaBoost[2] + probe_tn)
r2l_fpr = fp_AdaBoost[3]/ (fp_AdaBoost[3] + r2l_tn)
u2r_fpr = fp_AdaBoost[4]/ (fp_AdaBoost[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_AdaBoost = classification_report(test_Y, predict_y_using_AdaBoost)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_AdaBoost) + '\n')

#compute accuracy metric
accuracy_AdaBoost = accuracy_score(test_Y, predict_y_using_AdaBoost)
print('Accuracy Score = ' + str(accuracy_AdaBoost) + '\n')

#compute precision metric
precision_AdaBoost = precision_score(test_Y, predict_y_using_AdaBoost, average = 'weighted')
print('Precision Score = ' + str(precision_AdaBoost) + '\n')

#compute recall metric
recall_AdaBoost = recall_score(test_Y, predict_y_using_AdaBoost, average = 'weighted')
print('Recall Score = ' + str(recall_AdaBoost) + '\n')

#compute F1-score metric
f1_score_AdaBoost = f1_score(test_Y, predict_y_using_AdaBoost, average = 'weighted')
print('F1 Score = ' + str(f1_score_AdaBoost) + '\n')

#compute Total Model built time
AdaBoost_built_time = AdaBoost_end_time - AdaBoost_start_time
print('Model Built Time = ' + str(AdaBoost_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_AdaBoost = np.sum(fp_AdaBoost)/ (np.sum(fp_AdaBoost) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_AdaBoost) + '\n')



### Implementation of KNN (Accuracy Achieved = 0.77...) ###

from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(weights='uniform', n_neighbors=100)
#get model building start time
KNN_start_time = dt.datetime.now()
#fit data to classifier
KNN_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_KNN = KNN_classifier.predict(test_x)
#get model building end time
KNN_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_KNN = confusion_matrix(test_Y, predict_y_using_KNN)
error = zero_one_loss(test_Y, predict_y_using_KNN)

print('=' * 20)
print('|  KNN MODEL  |')
print('=' * 20)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_KNN) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_KNN_df = pd.DataFrame(confusion_matrix_from_KNN, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for KNN Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_KNN_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_KNN = confusion_matrix_from_KNN.astype('float') / confusion_matrix_from_KNN.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_KNN.diagonal()))

#Compute False Positive Rate per label
tp_KNN = np.diag(confusion_matrix_from_KNN)
print('True Positive Per Label: ' + str(tp_KNN))

fp_KNN = np.sum(confusion_matrix_from_KNN, axis=0) - tp_KNN
print('False Positive Per Label: ' + str(fp_KNN))

fn_KNN = np.sum(confusion_matrix_from_KNN, axis=1) - tp_KNN
print('False Negative Per Label: ' + str(fn_KNN))

benign_tn = np.sum(confusion_matrix_from_KNN) - (tp_KNN[0]+ fp_KNN[0] + fn_KNN[0])
dos_tn = np.sum(confusion_matrix_from_KNN) - (tp_KNN[1]+ fp_KNN[1] + fn_KNN[1])
probe_tn = np.sum(confusion_matrix_from_KNN) - (tp_KNN[2]+ fp_KNN[2] + fn_KNN[2])
r2l_tn = np.sum(confusion_matrix_from_KNN) - (tp_KNN[3]+ fp_KNN[3] + fn_KNN[3])
u2r_tn = np.sum(confusion_matrix_from_KNN) - (tp_KNN[4]+ fp_KNN[4] + fn_KNN[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_KNN[0]/ (fp_KNN[0] + benign_tn)
dos_fpr = fp_KNN[1]/ (fp_KNN[1] + dos_tn)
probe_fpr = fp_KNN[2]/ (fp_KNN[2] + probe_tn)
r2l_fpr = fp_KNN[3]/ (fp_KNN[3] + r2l_tn)
u2r_fpr = fp_KNN[4]/ (fp_KNN[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_KNN = classification_report(test_Y, predict_y_using_KNN)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_KNN) + '\n')

#compute accuracy metric
accuracy_KNN = accuracy_score(test_Y, predict_y_using_KNN)
print('Accuracy Score = ' + str(accuracy_KNN) + '\n')

#compute precision metric
precision_KNN = precision_score(test_Y, predict_y_using_KNN, average = 'weighted')
print('Precision Score = ' + str(precision_KNN) + '\n')

#compute recall metric
recall_KNN = recall_score(test_Y, predict_y_using_KNN, average = 'weighted')
print('Recall Score = ' + str(recall_KNN) + '\n')

#compute F1-score metric
f1_score_KNN = f1_score(test_Y, predict_y_using_KNN, average = 'weighted')
print('F1 Score = ' + str(f1_score_KNN) + '\n')

#compute Total Model built time
KNN_built_time = KNN_end_time - KNN_start_time
print('Model Built Time = ' + str(KNN_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_KNN = np.sum(fp_KNN)/ (np.sum(fp_KNN) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_KNN) + '\n')



### Implementation of Decision Tree Classifier 5-class classification version (Accuracy Achieved = 0.799...) ###

from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=90,splitter= 'random',criterion='gini',random_state=0)
#get model building start time
decision_tree_start_time = dt.datetime.now()
#fit data to classifier
decision_tree_classifier.fit(train_x, train_Y)
#prediction
predict_y_using_decision_tree = decision_tree_classifier.predict(test_x)
#get model building end time
decision_tree_end_time = dt.datetime.now()
#compute confusion matrix
confusion_matrix_from_decision_tree = confusion_matrix(test_Y, predict_y_using_decision_tree)
error = zero_one_loss(test_Y, predict_y_using_decision_tree)

print('=' * 25)
print('|  DECISION TREE MODEL  |')
print('=' * 25)

#compute confusion matrix
print('\n' + 'Confusion Matrix:' + '\n' + str(confusion_matrix_from_decision_tree) + '\n')

#plot confusion matrix heatmap;
print('Confusion Matrix Heatmap:')
confusion_matrix_from_decision_tree_df = pd.DataFrame(confusion_matrix_from_decision_tree, index = ['benign', 'dos', 'probe', 'r2l', 'u2r'], columns = ['benign', 'dos', 'probe', 'r2l', 'u2r'])
plt.figure(figsize=(15,10))
plt.title('Confusion Matrix for decision_tree Classifier on KDD-NSL Dataset')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
sn.heatmap(confusion_matrix_from_decision_tree_df, annot=True, annot_kws={"size": 10}) # font size
plt.show()

#compute accuracy per label
accuracy_per_label_decision_tree = confusion_matrix_from_decision_tree.astype('float') / confusion_matrix_from_decision_tree.sum(axis=1)[:, np.newaxis]
print('Accuracy per Label: ' + str(accuracy_per_label_decision_tree.diagonal()))

#Compute False Positive Rate per label
tp_decision_tree = np.diag(confusion_matrix_from_decision_tree)
print('True Positive Per Label: ' + str(tp_decision_tree))

fp_decision_tree = np.sum(confusion_matrix_from_decision_tree, axis=0) - tp_decision_tree
print('False Positive Per Label: ' + str(fp_decision_tree))

fn_decision_tree = np.sum(confusion_matrix_from_decision_tree, axis=1) - tp_decision_tree
print('False Negative Per Label: ' + str(fn_decision_tree))

benign_tn = np.sum(confusion_matrix_from_decision_tree) - (tp_decision_tree[0]+ fp_decision_tree[0] + fn_decision_tree[0])
dos_tn = np.sum(confusion_matrix_from_decision_tree) - (tp_decision_tree[1]+ fp_decision_tree[1] + fn_decision_tree[1])
probe_tn = np.sum(confusion_matrix_from_decision_tree) - (tp_decision_tree[2]+ fp_decision_tree[2] + fn_decision_tree[2])
r2l_tn = np.sum(confusion_matrix_from_decision_tree) - (tp_decision_tree[3]+ fp_decision_tree[3] + fn_decision_tree[3])
u2r_tn = np.sum(confusion_matrix_from_decision_tree) - (tp_decision_tree[4]+ fp_decision_tree[4] + fn_decision_tree[4])
print('True Negative Per Label: ' + str([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))

benign_fpr = fp_decision_tree[0]/ (fp_decision_tree[0] + benign_tn)
dos_fpr = fp_decision_tree[1]/ (fp_decision_tree[1] + dos_tn)
probe_fpr = fp_decision_tree[2]/ (fp_decision_tree[2] + probe_tn)
r2l_fpr = fp_decision_tree[3]/ (fp_decision_tree[3] + r2l_tn)
u2r_fpr = fp_decision_tree[4]/ (fp_decision_tree[4] + u2r_tn)
print('False Positive Rate Per Label: '+ str([benign_fpr, dos_fpr, probe_fpr, r2l_fpr, u2r_fpr]))

#compute classification report
classification_report_decision_tree = classification_report(test_Y, predict_y_using_decision_tree)
print('\n' + 'Classification Report:' + '\n' + str(classification_report_decision_tree) + '\n')

#compute accuracy metric
accuracy_decision_tree = accuracy_score(test_Y, predict_y_using_decision_tree)
print('Accuracy Score = ' + str(accuracy_decision_tree) + '\n')

#compute precision metric
precision_decision_tree = precision_score(test_Y, predict_y_using_decision_tree, average = 'weighted')
print('Precision Score = ' + str(precision_decision_tree) + '\n')

#compute recall metric
recall_decision_tree = recall_score(test_Y, predict_y_using_decision_tree, average = 'weighted')
print('Recall Score = ' + str(recall_decision_tree) + '\n')

#compute F1-score metric
f1_score_decision_tree = f1_score(test_Y, predict_y_using_decision_tree, average = 'weighted')
print('F1 Score = ' + str(f1_score_decision_tree) + '\n')

#compute Total Model built time
decision_tree_built_time = decision_tree_end_time - decision_tree_start_time
print('Model Built Time = ' + str(decision_tree_built_time) + '\n')

#compute overall False positive rate
overall_false_positive_rate_decision_tree = np.sum(fp_decision_tree)/ (np.sum(fp_decision_tree) +np.sum([benign_tn, dos_tn, probe_tn, r2l_tn, u2r_tn]))
print('Overall False Positive Rate = ' + str(overall_false_positive_rate_decision_tree) + '\n')


"""
Step 8: Visualising Overall Algorithms Implemented Summary

"""

#get score values for bar chart heights
accuracy_bars = [round(accuracy_gaussian_naives_bayes * 100, 2), round(accuracy_logistic_regression * 100, 2), round(accuracy_support_vector * 100, 2),
                round(accuracy_AdaBoost * 100, 2), round(accuracy_KNN * 100, 2), round(accuracy_decision_tree * 100, 2)]

precision_bars = [round(precision_gaussian_naives_bayes * 100, 2), round(precision_logistic_regression * 100, 2), round(precision_support_vector * 100, 2),
                round(precision_AdaBoost * 100, 2), round(precision_KNN * 100, 2), round(precision_decision_tree * 100, 2)]

recall_bars = [round(recall_gaussian_naives_bayes * 100, 2), round(recall_logistic_regression * 100, 2), round(recall_support_vector * 100, 2),
                round(recall_AdaBoost * 100, 2), round(recall_KNN * 100, 2), round(recall_decision_tree * 100, 2)]

f1_score_bars = [round(f1_score_gaussian_naives_bayes * 100, 2), round(f1_score_logistic_regression * 100, 2), round(f1_score_support_vector * 100, 2),
                round(f1_score_AdaBoost * 100, 2), round(f1_score_KNN * 100, 2), round(f1_score_decision_tree * 100, 2)]

overall_false_positive_rate_bars = [round(overall_false_positive_rate_gaussian_naive_bayes * 100, 2), round(overall_false_positive_rate_logistic_regression * 100, 2), round(overall_false_positive_rate_support_vector * 100, 2),
                                    round(overall_false_positive_rate_AdaBoost * 100, 2), round(overall_false_positive_rate_KNN * 100, 2), round(overall_false_positive_rate_decision_tree * 100, 2)]

#set bar size
bar_size = 0.10

#bar positioning
position_accuracy = np.arange(len(accuracy_bars))
position_precision = [value + bar_size for value in position_accuracy]
position_recall = [value + bar_size for value in position_precision]
position_f1_score = [value + bar_size for value in position_recall]
position_overall_false_positive_rate = [value + bar_size for value in position_f1_score]

plt.figure(figsize=(25,10))

#plot bars
plt.bar(position_accuracy, accuracy_bars, color = 'red', width = bar_size, label='Accuracy Score')
plt.bar(position_precision, precision_bars, color = 'blue', width = bar_size, label='Precision Score')
plt.bar(position_recall, recall_bars, color = 'yellow', width = bar_size, label='Recall Score')
plt.bar(position_f1_score, f1_score_bars, color = 'black', width = bar_size, label='F1 Score')
plt.bar(position_overall_false_positive_rate, overall_false_positive_rate_bars, color = 'green', width = bar_size, label='False Positive Rate Score')

#Adding Group labels and X-axis labels
# Add xticks on the middle of the group bars
plt.title('BAR CHART ILLUSTRATING SCORES OF EACH ALGORITHM IMPLEMENTED')
plt.xlabel('MACHINE LEARNING CLASSIFIERS', fontweight='bold')
plt.ylabel('%', fontweight='bold')
plt.xticks([algorithms + bar_size for algorithms in range(len(accuracy_bars))], ['Gaussian Naive Bayes', 'Logistic Regression', 'Support Vector Machine', 'AdaBoost', 
            'K Nearest Neighbour', 'Decision Tree'])

#Display barchart and legend
plt.legend()
plt.show()
