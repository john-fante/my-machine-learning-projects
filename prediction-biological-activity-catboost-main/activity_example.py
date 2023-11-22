# -*- coding: utf-8 -*-
"""activity_example.ipynb

# Prediction Biological Activity with CatBoost Implementation

*The main goal of this project is prediction biological activity for inhibiting GPCR-mediated pheromone signaling in the invasive sea lamprey using the dataset created from the ligand-based screening dataset.*


# Results

- Matthews Correlation Coefficient(MCC) : 0.8257228
- Test Accuracy score : 92.857%

# References

- Raschka, S., Scott, A.M., Huertas, M., Li, W., Kuhn, L.A. (2018). Automated Inference of Chemical Discriminants of Biological Activity. In: Gore, M., Jagtap, U. (eds) Computational Drug Discovery and Design. Methods in Molecular Biology, vol 1762. Humana Press, New York, NY. https://doi.org/10.1007/978-1-4939-7756-7_16

- original code and data -> https://github.com/psa-lab/predicting-activity-by-machine-learning
"""

!pip install catboost
!pip install ppscore

# importing dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, cv, Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import ppscore as pps

data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GeneralDB/dkpes.csv')
data.head()

# I have applied to the mutual information due to find the most valuable features

numeric_data = data.copy()
numeric_data.drop('ShapeQuery', axis = 1, inplace = True)

mi = mutual_info_regression(numeric_data.iloc[:,2:-1], y = numeric_data['Signal-inhibition'] )

mi = pd.DataFrame(mi, columns = ["miScore"] ,index = list(numeric_data.iloc[:,2:-1].columns))
mi[mi['miScore'] > 0.12].sort_values(by = 'miScore', ascending = False)

# 1 active, 0 non-active, the referance is the paper.

def create_cat_func(X):
  if X <= 0.6:
    return 0
  else:
    return 1

data['Signal-inhibition'] = data['Signal-inhibition'].apply(create_cat_func)

y = data['Signal-inhibition']
X = data.drop(['Signal-inhibition'], axis = 1)

X.head()

# define cat features

cat_feat_idx =  np.where(X.dtypes == 'object')[0]
cat_feat_idx

# class distributions

y.value_counts()

# train-test splitting

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, stratify= y, random_state=0)

# creating the model and training

model = CatBoostClassifier(iterations = 1000, cat_features = cat_feat_idx, verbose = 250, max_depth = 5)

model.fit(X_train, y_train)

# predicting and calculating some classification metrics (mcc, acc etc.)

pred = model.predict(X_test)

report = classification_report(y_test, pred)
matthews_Coef = matthews_corrcoef(y_test, pred)
acc_scr = accuracy_score(y_test, pred)*100

print("matthews_Coef :" + str(matthews_Coef))
print('accuracy score:{0:.3f}%'.format(acc_scr))
print(report)

# plotting confusion matrix

cm =  confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(cm, display_labels=['non-active','active'])
disp.plot()

# plotting the first tree

model.plot_tree(tree_idx = 0)

# catboosting cross-validation wiht 6 folds

cv_pool = Pool(X, y, cat_features = cat_feat_idx )


cv_params = {
    "iterations" : 1000,
    "loss_function": "CrossEntropy",
    "verbose": 250
    }


CV_scores = cv(cv_pool, params = cv_params, fold_count =6 )

CV_scores.sort_values(by = "test-CrossEntropy-mean").head()