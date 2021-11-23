# In this chapter, you will learn about three fundamental techniques that will help us to summarize the information content of a dataset by transforming it onto a new feature subspace of lower dimensionality than the original one.

## Principal component analysis (PCA) for unsupervised data compression
## Linear Discriminant Analysis (LDA) as a supervised dimensionality reduction technique for maximizing class separability
## Nonlinear dimensionality reduction via kernel principal component analysis

######## PRINCIPAL COMPONENT ANALYSIS (PCA) ########
import pandas as pd

df_wine = pd.read_csv('/Users/WIBE/Downloads/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
'Alcalinity of ash', 'Magnesium', 'Total phenols',
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


import numpy as np


cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)
