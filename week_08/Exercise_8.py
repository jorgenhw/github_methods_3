# Author: Jørgen Højlund Wibe (AU ID: 201807750)
# Methods 3 - portfolio 3/4

#################################################
################# EXERCISE 1 ####################
#################################################

# 1. Importing the libraries
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1.1: Loading data
data = np.load("/Users/WIBE/Desktop/cogSci/Methods_3/Git/github_methods_3/week_08/data/megmag_data.npy")

#1.1.1: How many repetitions, sensors and time samples are there?
data.shape # Output: (682, 102, 251)
"""
3 dimensions (axis) with length 682 (number of repetitions), 102 (number of sensors) and 251 (number of time samples).
"""

# 1.1.2: make a vector from -200-800 with a 4 jump between each
times = np.arange(-200, 801, 4)

print(times)

# 1.1.3: Making a sensor covariance matrix
cov_mat = [] # Creating an empty list

# calculating the dot product for all rows i using all data points in the dimensions
for i in range(len(data[:, 0, 0])):
    cov_mat.append(data[i,:,:] @ data[i,:,:].T) # @ = * (times) when dealing with matrices, append() appends an element to the end of the list, .T is command for transpose


# out of the loop the dot product of the the matrices for each i is summed and divided by n
cov_mat = sum(cov_mat)/len(data[:, 0, 0]) #divided by length of repetitions (682)

# plotting the covariance matrix
plt.imshow(cov_mat, cmap="hot") # n equal number of targets in each folds
plt.show() # show the one just made

# 1.1.4 Make an average over the repetition dimension using np.mean - use the axis argument. (The resulting array should have two dimensions with time as the first and magnetic field as the second)
avr_rep = np.mean(data, axis = 0) # the axis = 0 makes the function average axis 1 and 2 over axis 0, and thus we end up with a data set only consisting of sensors and times (2D df).

# for inspection
avr_rep.shape

# 1.1.5: Plot the magnetic field (based on the average) as it evolves over time for each of the sensors (a line for each) (time on the x-axis and magnetic field on the y-axis). Add a horizontal line at y = 0 and a vertical line at x = 0 using plt.axvline and plt.axhline
plt.figure()
plt.plot(times, avr_rep.T)
plt.axvline()
plt.axhline()
plt.ylabel("Magnetic field")
plt.xlabel("Time")
plt.title("SexyPlot")
plt.show()

# 1.1.6: Find the maximal magnetic field in the average. Then use np.argmax and np.unravel_index to find the sensor that has the maximal magnetic field.
max = np.unravel_index(np.argmax(avr_rep, axis=None), avr_rep.shape)
max # sensor 73 at repetition 112
print(max)

# 1.1.7: Plot the magnetic field for each of the repetitions (a line for each) for the sensor that has the maximal magnetic field. Highlight the time point with the maximal magnetic field in the average (as found in 1.1.v) using plt.axvline

# Plot the magnetic field for each of the repetitions (a line for each) for the sensor that has the maximal magnetic field (sensor 73)
plt.figure()
plt.plot(times, data[:, 73, :].T) # Taking all repetitions (axis0), only for sensor 73 (72 bcause of 0 index)
plt.axvline(x = 0)  # Highlight the time point with the maximal magnetic field in the average
plt.axhline(np.amax(data[:,73,:]))
plt.xlabel(" time")
plt.ylabel("magnetic field")
plt.title("magnetic field (based on the average)")
plt.show()

# 1.1.8: Describe in your own words how the response found in the average is represented in the single repetitions. But do make sure to use the concepts signal and noise and comment on any differences on the range of values on the y-axis
"""

"""

# 1.2 Load new data
y = np.load("/Users/WIBE/Desktop/cogSci/Methods_3/Git/github_methods_3/week_08/data/pas_vector.npy")


# 1.2.1
y.shape # Output: (682,)
# this data set has the same length as the number of repetitions from the earlier data set.


# 1.2.2 Now make four averages (As in Exercise 1.1.3), one for each PAS rating, and plot the four time courses (one for each PAS rating) for the sensor found in Exercise 1.1.v

# Making empty lists for each PAS rating
pas_1 = []
pas_2 = []
pas_3 = []
pas_4 = []

# Making a new data set with values only coming from sensor 73
sen73 = data[:, 73, :]

# Adding together the indexed numbers belonging to each PAS value (so I can later join the sen73 data with this index number)
for i in range(len(y)):
    if y[i] == 1:
        pas_1.append(i)
    if y[i] == 2:
        pas_2.append(i)
    if y[i] == 3:
        pas_3.append(i)
    if y[i] == 4:
        pas_4.append(i)

# calculating the average
avr_rep_pas1 = np.mean(sen73[pas_1], axis=0)
avr_rep_pas2 = np.mean(sen73[pas_2], axis=0)
avr_rep_pas3 = np.mean(sen73[pas_3], axis=0)
avr_rep_pas4 = np.mean(sen73[pas_4], axis=0)

# Plotting
plt.figure()
plt.plot(times, avr_rep_pas1, label="PAS 1")
plt.plot(times, avr_rep_pas2, label="PAS 2")
plt.plot(times, avr_rep_pas3, label="PAS 3")
plt.plot(times, avr_rep_pas4, label="PAS 4")
plt.xlabel("Time")
plt.ylabel("Magnetic field / brain activity")
plt.title("PAS ratings for sensor 73")
plt.legend(loc="upper left")
plt.show()


# 1.2.3 Notice that there are two early peaks (measuring visual activity from the brain), one before 200 ms and one around 250 ms. Describe how the amplitudes of responses are related to the four PAS-scores. Does PAS 2 behave differently than expected?
"""
In general participants who rated PAS2 had a slightly higher activity in their visual cortex(?). If this effect is significant is hard to tell, but since the lines are averages across many repetitions one could imagine that the effect is significant.
Also when looking at the four lines, there seems to be a systematic pattern as to how the brain activity differs depending on which PAS rating the participants reported. If PAS4 was reported, the brain activity from 150-250ms was generally lower than if they rated pas 3, 2 or 1. The PAS2 ratings in particular are interesting, though, since the seem to deviate quite a lot from this pattern. On reason could be, that the participants could actually sense something but were a bit unsure and thus used extra cognitive resources to figure out what they saw. For PAS1 the target might have been too blurred for the participants even to worry about trying to figure out what was behind. 
"""

#################################################
################# EXERCISE 2 ####################
#################################################

# 2.1.1: We’ll start with a binary problem - create a new array called data_1_2 that only contains PAS responses 1 and 2. Similarly, create a y_1_2 for the target vector

y_1_2 = []
for i in range(len(y)):
    if y[i] == 1:
        y_1_2.append(1)
    if y[i] == 2:
        y_1_2.append(2)

data_1_2 = np.concatenate((data[pas_1], data[pas_2]), axis=0) # 3D

# 2.1.2: Scikit-learn expects our observations (data_1_2) to be in a 2d-array, which has samples (repetitions) on dimension 1 and features (predictor variables) on dimension 2. Our data_1_2 is a three-dimensional array. Our strategy will be to collapse our two last dimensions (sensors and time) into one dimension, while keeping the first dimension as it is (repetitions). Use np.reshape to create a variable X_1_2 that fulfils these criteria.
## Answer to Q: reshape(3,1) first number; the number of dimensions that I want.
## We cant really interpret the flattened data frame, but we need to flatten it in order for sklearn to be able to work with it.

# repetition as rows, and sensor and time as columns
X_1_2 = data_1_2.reshape(214, 102*251)

# 2.1.3: Import the StandardScaler and scale X_1_2
from sklearn.preprocessing import StandardScaler # package to standardize values in df

sc = StandardScaler()
X_1_2_scaled = sc.fit_transform(X_1_2)

# 2.1.4: Do a standard LogisticRegression - can be imported from sklearn.linear_model - make sure there is no penalty applied
from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(penalty='none') # no regularisation

logR.fit(X_1_2_scaled, y_1_2)

# 2.1.5: Use the score method of LogisticRegression to find out how many labels were classified correctly. Are we overfitting? Besides the score, what would make you suspect that we are over fitting?

print(logR.score(X_1_2_scaled, y_1_2))

"""
Besides the score, the fact that we are not penalizing the model is a clear indication that we risk overfitting. On top of that we are no dividing the data into test and training which means that our model just memorizes the data.
"""

# 2.1.6: Now apply the L1 penalty instead - how many of the coefficients (.coef_) are non-zero after this?
logR = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=1) # With regularization
logR.fit(X_1_2_scaled, y_1_2)
print(logR.score(X_1_2_scaled, y_1_2))

fit1 = logR.fit(X_1_2_scaled, y_1_2)

print(np.sum(fit1.coef_ == 0))
print(np.sum(fit1.coef_ != 0)) # = 217 coefs were nonzero

# 2.1.7: Create a new reduced X that only includes the non-zero coefficients - show the covariance of the non-zero features (two covariance matrices can be made; X_reducedXT or XT Xreduced (you choose the right one)) . Plot the covariance of the features using plt.imshow. Compared to the plot from 1.1.iii, do we see less covariance?
coefs = fit1.coef_.flatten()
non_zero = coefs != 0

X_reduced = X_1_2_scaled[:, non_zero]

covmat = X_reduced.T @ X_reduced

import matplotlib.pyplot as plt
plt.imshow(covmat)
plt.show()

# 2.2: Now, we are going to build better (more predictive) models by using cross-validation as an outcome measure

# 2.2.1: Import cross_val_score and StratifiedKFold from sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# 2.2.2: To make sure that our training data sets are not biased to one target (PAS) or the other, create y_1_2_equal, which should have an equal number of each target. Create a similar X_1_2_equal. The function equalize_targets_binary in the code chunk associated with Exercise 2.2.ii can be used. Remember to scale X_1_2_equal!

def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
# randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
# create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    return new_data, new_y

y_1_2 = np.array(y_1_2)



X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)

# reshape X
X_1_2_equal = X_1_2_equal.reshape(198, 102*251)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_1_2_equal_std = sc.fit_transform(X_1_2_equal)


# 2.2.3 Do cross-validation with 5 stratified folds doing standard LogisticRegression (See Exercise 2.1.iv)
## Start by making a logistic regression (with the new data)
from sklearn.linear_model import LogisticRegression
logR = LogisticRegression(penalty='none') # no regularisation
logR.fit(X_1_2_equal_std, y_1_2_equal)

# Then run over it

# Generate test sets such that all contain the same distribution of classes, or as close as possible.
cv = StratifiedKFold() # default cv is 5 fold stratification

scores = cross_val_score(logR, X_1_2_equal_std, y_1_2_equal, cv=cv)
print(np.mean(scores)) # output = 0.4746

# 2.2.4 Do L2-regularisation with the following Cs= [1e5, 1e1, 1e-5]. Use the same kind of cross-validation as in Exercise 2.2.iii. In the best-scoring of these models, how many more/fewer predictions are correct (on average)?

# regression 1
logR = LogisticRegression(C=1e5, penalty="l2", solver='liblinear', random_state=1) # With ridge (l2) regularization
logR.fit(X_1_2_equal_std, y_1_2_equal) # Fitting model
scores = cross_val_score(logR, X_1_2_equal_std, y_1_2_equal, cv=cv) # Adding 5-fold stratified kFold
print(np.mean(scores)) # Output = 0.4698

# regression 2
logR = LogisticRegression(C=1e1, penalty="l2", solver='liblinear', random_state=1) # With ridge (l2) regularization
logR.fit(X_1_2_equal_std, y_1_2_equal)

scores = cross_val_score(logR, X_1_2_equal_std, y_1_2_equal, cv=cv)
print(np.mean(scores)) # output = 0.47

# regression 3
logR = LogisticRegression(C=1e-5, penalty="l2", solver='liblinear', random_state=1) # With regularization
logR.fit(X_1_2_equal_std, y_1_2_equal)

scores = cross_val_score(logR, X_1_2_equal_std, y_1_2_equal, cv=cv)
print(np.mean(scores)) # output = 0.48

# 2.2.5: Instead of fitting a model on all n_sensors * n_samples features, fit a logistic regression (same kind as in Exercise 2.2.iv (use the C that resulted in the best prediction)) for each time sample and use the same cross-validation as in Exercise 2.2.iii. What are the time points where classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)

def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
# randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
# create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    return new_data, new_y

data_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2) # Applying function to data and target variable

output = [] # Making an empty data variable

# subsetting time
for i in range(251): #251 = number of time scales
    std = sc.fit_transform(data_1_2_equal[:, :, i])
    logR.fit(std, y_1_2_equal)
    scores = cross_val_score(logR, std, y_1_2_equal, cv=5)
    output.append(np.mean(scores))

# Best classifications
np.amax(output)
np.argmax(output)

# plotting
plt.figure()
plt.plot(times, output)
plt.axvline(x = 0)
plt.axhline(y = 0.5)
plt.xlabel("Time")
plt.ylabel("classification accuracy")
plt.title("plåt")
plt.show()

# 2.2.6: Now do the same, but with L1 regression - set C=1e-1 - what are the time points when classification is best (make a plot)?

# regression
logR = LogisticRegression(C=1e-1, penalty="l1", solver='liblinear', random_state=1) # with lasso (l1) regularization

output_2 = [] #making an empty list for the loop

# subsetting time
for i in range(len(times)): #251 = number of time scales
    std = sc.fit_transform(data_1_2_equal[:, :, i])
    logR.fit(std, y_1_2_equal)
    scores = cross_val_score(logR, std, y_1_2_equal, cv=5)
    output_2.append(np.mean(scores))

# Best classifications
np.amax(output_2)
np.argmax(output_2)

# plotting
plt.figure()
plt.plot(times, output_2)
plt.axvline(x = 0)
plt.axhline(y = 0.5)
plt.xlabel("Time")
plt.ylabel("classification accuracy")
plt.title("plot 2")
plt.show()

# 2.2.7: Finally, fit the same models as in Exercise 2.2.6 but now for data_1_4 and y_1_4 (create a data set and a target vector that only contains PAS responses 1 and 4). What are the time points when classification is best? Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)


# 2.2.7.1: Create a data set and a target vector that only contains PAS responses 1 and 4
y_1_4 = []
for i in range(len(y)):
    if y[i] == 1:
        y_1_4.append(1)
    if y[i] == 4:
        y_1_4.append(4)

data_1_4 = np.concatenate((data[pas_1], data[pas_4]), axis=0) # 3D

# 2.2.7.2: Fit the same models as in Exercise 2.2.6 but now for data_1_4 and y_1_4

# 2.2.7.2.1: Equalizing data
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
# randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
# create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    return new_data, new_y

#turning y into an array for the func to function
y_1_4 = np.array(y_1_4)

# accessing function
data_1_4_equal, y_1_4_equal = equalize_targets_binary(data_1_4, y_1_4) # Applying function to data and target variable

# Defining parameters of regression
logR = LogisticRegression(C=1e-1, penalty="l1", solver='liblinear', random_state=1) # with lasso (l1) regularization

output_3 = [] #making an empty list for the loop

# subsetting time
for i in range(len(times)): #251 = number of time scales
    std = sc.fit_transform(data_1_4_equal[:, :, i])
    logR.fit(std, y_1_4_equal)
    scores = cross_val_score(logR, std, y_1_4_equal, cv=5)
    output_3.append(np.mean(scores))

# 2.2.7.3: What are the time points when classification is best?
print(np.amax(output_3)) # 0.585
print(np.argmax(output_3)) # 118


# 2.2.7.4: Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)
plt.figure()
plt.plot(times, output_3)
plt.axvline(x = 0)
plt.axhline(y = 0.5)
plt.xlabel("Time")
plt.ylabel("classification accuracy")
plt.title("Plot 3")
plt.show()

##############################################
# What is the chance level for this analysis?
##############################################


# 2.3: Is pairwise classification of subjective experience possible? Any surprises in the classification accuracies, i.e. how does the classification score fore PAS 1 vs 4 compare to the classification score for PAS 1 vs 2?

"""
Generally it is complicated to make classifications based on subjective experiences. 
"""

#################################################
################# EXERCISE 3 ####################
#################################################

# 3.1: Do a Support Vector Machine Classification

# 3.1.1:  First equalize the number of targets using the function associated with each PAS-rating using the function associated with Exercise 3.1.1

# Equalizer function
def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)

    new_indices = np.concatenate((first_choice, second_choice, third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    return new_data, new_y

# Running function on data
(data_equal, y_equal) = equalize_targets(data, y)


# 3.1.2 Run two classifiers, one with a linear kernel and one with a radial basis (other options should be left at their defaults) - the number of features is the number of sensors multiplied the number of samples. Which one is better predicting the category?
cv = StratifiedKFold() # cross validation

# Making data from 3D to 2D
data_equal_2d = data_equal.reshape(396, 102*251)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(data_equal_2d)
X_std = sc.transform(data_equal_2d)
from sklearn.svm import SVC

# Linear kernel
svm = SVC(kernel='linear')

scores_svm = cross_val_score(svm, X_std, y_equal, cv=cv)
print(np.mean(scores_svm)) # = 0.2928164556962025 (accuracy of classification (29%), just above chance level of 25%)

# Radial kernel
svm = SVC(kernel='rbf')

scores_svm = cross_val_score(svm, X_std, y_equal, cv=cv)
print(np.mean(scores_svm)) # = 0.3333544303797468


# 3.1.3 Run the sample-by-sample analysis (similar to Exercise 2.2.5) with the best kernel (from Exercise 3.1.2). Make a plot with time on the x-axis and classification score on the y-axis with a horizontal line at the chance level (what is the chance level for this analysis?)

output = [] # Making an empty data variable

# Defining number of kfolds
cv = StratifiedKFold(n_splits=5)

# subsetting time
for i in range(251): #251 = number of time scales
    scaler = StandardScaler()
    X_time = data_equal[:, :, i] # standardizing
    X_time_reshaped = X_time.reshape(X_time.shape[0], -1)  # transforming
    X_time_scaled = scaler.fit_transform(X_time_reshaped)
    # Making a support vector machine with radial as kernel
    sup_mach_radial = SVC(kernel = 'rbf')
    # Cross validating
    score = cross_val_score(sup_mach_radial, X_time_scaled, y_equal, cv=cv)
    # taking the mean
    mean = np.mean(score)
    # appending the mean
    output.append(mean)


# Best classifications
np.amax(output)
np.argmax(output)

# plotting
plt.figure()
plt.plot(times, output)
plt.axhline(y = 0.25)
plt.xlabel("Time")
plt.ylabel("classification accuracy")
plt.title("plåt")
plt.show()

# 3.1.4: Is classification of subjective experience possible at around 200-250 ms?
"""
The classification accuracy is consistently above chance level from 200 - 250 ms meaning, classification of subjective experiences are indeed possible.
"""

# 3.2: Finally, split the equalized data set (with all four ratings) into a training part and test part, where the test part if 30 % of the trials. Use train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_equal, y_equal, test_size=0.30, random_state=42)

# 3.2.1: Use the kernel that resulted in the best classification in Exercise 3.1.2 and fit the training set and predict on the test set. This time your features are the number of sensors multiplied by the number of samples.
# As the radial kernel performed the best, I use that one:
radial_kernel = SVC(kernel = 'rbf')

# Rehshaping test and train set
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

# Fitting
radial_kernel.fit(X_train_reshaped, y_train)

## predicting
predicted_y = radial_kernel.predict(X_test_reshaped)

acc = list(y_test == predicted_y)

## Proportion of correctly predicted pas-scores
print("Proportion of correctly predicted pas:", round(acc.count(True)/len(acc), 3)) # = 0.252
"""
Performance is insuaccfficint for actual use.
"""

# 3.2.2: Create a confusion matrix. It is a 4x4 matrix. The row names and the column names are the PAS-scores. There will thus be 16 entries. The PAS1xPAS1 entry will be the number of actual PAS1, ypas1 that were predicted as PAS1, yˆpas1. The PAS1xPAS2 entry will be the number of actual PAS1, ypas1 that were predicted as PAS2, yˆpas2 and so on for the remaining 14 entries. Plot the matrix
import pandas as pd

predicted_y = pd.Series(predicted_y, name = 'Predicted PAS')
y_test = pd.Series(y_test, name = 'Actual PAS')
print(pd.crosstab(y_test, predicted_y))
# Output:
"""
Predicted PAS  1  2   3   4
Actual PAS                 
1              1  5  19  12
2              2  5  13   8
3              0  2  15   9
4              2  3  14   9
"""

# 3.2.3: Based on the confusion matrix, describe how ratings are misclassified and if that makes sense given that ratings should measure the strength/quality of the subjective experience. Is the classifier biased towards specific ratings?
print("Pas1 Misclassification:", round((5+19+12)/(1+5+19+12)*100, 3)) #pas1 = 97.297
print("Pas2 Misclassification:", round((2+13+8)/(2+5+13+8)*100, 3)) #pas2 = 82.143
print("Pas3 Misclassification:", round((0+2+9)/(0+2+15+9)*100, 3)) #pas3 = 42.308
print("Pas4 Misclassification:", round((2+3+14)/(2+3+14+9)*100, 3)) #pas4 = 67.857

"""
The seems to be a bias towards classifying pas rating as three (assessed by looking at the 3rd column). 
"""
