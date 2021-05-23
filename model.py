# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:35:59 2021

@author: Apoorv Jain
"""

#Implementing XGBoosting

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold


#%% Creating pre defined functions    
#%% create a generalized function to calculate the metrics values for test set
def get_test_report(model):
    
    # for test set:
    # test_pred: prediction made by the model on the test dataset 'X_test'
    # y_test: actual values of the target variable for the test dataset

    # predict the output of the target variable from the test data 
    test_pred = model.predict(X_test)

    # return the classification report for test data
    return(classification_report(y_test, test_pred))

#%% Confusion Matrix
def plot_confusion_matrix(model):
    y_pred = model.predict(X_test)
    
    # create a confusion matrix
    # pass the actual and predicted target values to the confusion_matrix()
    cm = confusion_matrix(y_test, y_pred)

    # label the confusion matrix  
    # pass the matrix as 'data'
    # pass the required column names to the parameter, 'columns'
    # pass the required row names to the parameter, 'index'
    conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:Low','Predicted:Good','Predicted:Average','Predicted:High'], index = ['Actual:Low','Actual:Good','Actual:Average','Actual:High'])

    # plot a heatmap to visualize the confusion matrix
    # 'annot' prints the value of each grid 
    # 'fmt = d' returns the integer value in each grid
    # 'cmap' assigns color to each grid
    # as we do not require different colors for each grid in the heatmap,
    # use 'ListedColormap' to assign the specified color to the grid
    # 'cbar = False' will not return the color bar to the right side of the heatmap
    # 'linewidths' assigns the width to the line that divides each grid
    # 'annot_kws = {'size':25})' assigns the font size of the annotated text 
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cbar = False, 
                linewidths = 0.1, annot_kws = {'size':25})

    # set the font size of x-axis ticks using 'fontsize'
    plt.xticks(rotation=90,fontsize = 20)

    # set the font size of y-axis ticks using 'fontsize'
    plt.yticks(rotation=0,fontsize = 20)

    # display the plot
    plt.show()
    
#%% Loading the dataset

file_path = 'G:/DSE/Project/Dataset/StudentSalary/df_final1.csv'
df_model = pd.read_csv(file_path)


X=df_model.iloc[:,1:32]
y= df_model.iloc[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)



xgb= XGBClassifier(use_label_encoder=True)

model= xgb.fit(X_train,y_train)

print(model)

y_pred = xgb.predict(X_test)
accuracy_score(y_test,y_pred)

y.value_counts()


plot_confusion_matrix(model)

print(get_test_report(model))


#%% Hyperparameter Tuning

tuning_parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     'max_depth': range(3,10),
                     'gamma': [0, 1, 2, 3, 4,5]}

xgb_model = XGBClassifier(use_label_encoder=True)
kfold= StratifiedKFold(n_splits=10,random_state=7)

xgb_grid = GridSearchCV(estimator = xgb_model, param_grid = tuning_parameters, cv = kfold)

xgb_grid.fit(X_train, y_train)

print('Best parameters for XGBoost classifier: ', xgb_grid.best_params_, '\n')


#%% bulding model on tuned parameters

xgb_grid_model = XGBClassifier(learning_rate = xgb_grid.best_params_.get('learning_rate'),
                               max_depth = xgb_grid.best_params_.get('max_depth'),
                               gamma = xgb_grid.best_params_.get('gamma'))

# use fit() to fit the model on the train set
xgb_model = xgb_grid_model.fit(X_train, y_train)

y_pred_new = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred_new)

# print the performance measures for test set for the model with best parameters
print('Classification Report for test set:\n', get_test_report(xgb_model))
plot_confusion_matrix(xgb_model)


#%%

important_features = pd.DataFrame({'Features': X_train.columns, 
                                   'Importance': xgb_model.feature_importances_})


important_features = important_features.sort_values('Importance', ascending = False)


sns.barplot(x = 'Importance', y = 'Features', data = important_features)


plt.title('Feature Importance', fontsize = 15)
plt.xlabel('Importance', fontsize = 15)
plt.ylabel('Features', fontsize = 15)
plt.show()

#%% 

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
df_model.skew()

df_transformed=pt.fit_transform(df_model.iloc[:,[3,9,13,14,16,28]])
np.where(df_model.skew()<-3)
np.where(df_model.skew()>3)

df_trans= pd.DataFrame(data=df_transformed,columns=df_model.iloc[:,[3,9,13,14,16,28]].columns)

    
