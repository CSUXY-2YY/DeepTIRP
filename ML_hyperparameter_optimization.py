# %%
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.ensemble import AdaBoostClassifier 
import lightgbm as lgb
from sklearn.metrics import roc_auc_score 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


# %%
pipe_steps = [
    ('svc', SVC()),
    ('lr', LogisticRegression()),    
    ('rb', RandomForestClassifier()),     
    ('knn', KNeighborsClassifier()),       
    ('gbc', GradientBoostingClassifier()),       
    ('xgb', XGBClassifier()),       
    ('abc', AdaBoostClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('lgb', lgb.LGBMClassifier())
    ]

param_grid = {    
    'rb__max_depth' : range(5,100,5),
    'rb__min_samples_split': range(5,100,5),
    'rb__min_samples_leaf' : range(5,100,5),
    'rb__bootstrap' : [True],
    'rb__criterion' : ["gini", "entropy"],
    'rb__n_estimators' : [50,100,150,200,300],

    'lr__penalty' : ['l1','l2'],
    'lr__solver' : ['liblinear'],
    'lr__C' : np.linspace(1, 1000, 10),   
    
    'svc__kernel' : ['rbf','linear'],
    'svc__gamma' : 2.0 ** np.arange(-10, 4),
    'svc__C' : [0.01, 0.1, 1, 10, 50, 100, 1000],   
    
    'knn__n_neighbors' : range(1,30,2),
    'knn__weights' : ['uniform', 'distance'],

    'dt__max_depth' : range(5,100,5),
    'dt__criterion' : ['entropy','gini'],    
    'dt__min_samples_split' : range(2,50,2),    
    'dt__min_samples_leaf' : range(2,50,2),     

    'gbc__loss' : ["deviance",'exponential'],
    'gbc__learning_rate' : [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],    
    'gbc__min_samples_split' : np.linspace(0.1, 0.5, 12),    
    'gbc__min_samples_leaf' : np.linspace(0.1, 0.5, 12),  
    'gbc__max_depth' : [3,5,8,10,20],
    'gbc__max_features' : ["log2","sqrt"],
    'gbc__criterion' : ["friedman_mse", "mae"],    
    'gbc__subsample' : [0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0],    
    'gbc__n_estimators' : [10,50,100,150,200],      
    
    'abc__n_estimators' : range(10,1000,10),
    'abc__learning_rate' : [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],   
    
    'lgb__max_depth' : [4,6,8],
    'lgb__num_leaves' : [20,30,40],    
    'lgb__min_child_samples' : [18,19,20,21,22],    
    'lgb__min_child_weight' : [0.001,0.002],  
    'lgb__feature_fraction' : [0.6, 0.8, 1],
    'lgb__bagging_fraction' : [0.8,0.9,1],
    'lgb__bagging_freq' : [2,3,4],    
    'lgb__cat_smooth' : [0,10,20],    
    'lgb__n_estimators' : [10,50,100,150,200],     
    
    'xgb__n_estimators' : [50,100,150,200,300],
    'xgb__eta' : [0.05, 0.1, 0,2, 0.3],    
    'xgb__max_depth' : [3,4,5,6,7],    
    'xgb__colsample_bytree' : [0.4,0.6,0.8,1],  
    'xgb__min_child_weight' : [1,2,3,4],
}

param_grid_dict = {
    
    'rb':{'rb__max_depth' : range(5,100,5),
    'rb__min_samples_split': range(5,100,5),
    'rb__min_samples_leaf' : range(5,100,5),
    'rb__bootstrap' : [True],
    'rb__criterion' : ["gini", "entropy"],
    'rb__n_estimators' : [50,100,150,200,300]}, 
                 
    # 'lr':{'lr__penalty' : ['l1','l2'],
    # 'lr__solver' : ['liblinear'],
    # 'lr__C' : np.linspace(1, 1000, 10)},

    'lr':{'lr__penalty' : ['l2'],
    'lr__solver' : ['liblinear','sag','newton-cg','lbfgs'],
    'lr__C' : list(np.linspace(0.05,1,19))},   
    
    'svc':{'svc__kernel' : ['rbf','linear'],
    'svc__gamma' : 2.0 ** np.arange(-10, 4),
    'svc__C' : [0.01, 0.1, 1, 10, 50, 100, 1000]}, 
    
    'knn':{'knn__n_neighbors' : range(1,30,2),
    'knn__weights' : ['uniform', 'distance']},

    'dt':{'dt__max_depth' : range(5,100,5),
    'dt__criterion' : ['entropy','gini'],    
    'dt__min_samples_split' : range(2,50,2),    
    'dt__min_samples_leaf' : range(2,50,2)},   

    'gbc':{'gbc__loss' : ["deviance",'exponential'],
    'gbc__learning_rate' : [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],    
    'gbc__min_samples_split' : np.linspace(0.1, 0.5, 12),    
    'gbc__min_samples_leaf' : np.linspace(0.1, 0.5, 12),  
    'gbc__max_depth' : [3,5,8,10,20],
    'gbc__max_features' : ["log2","sqrt"],
    'gbc__criterion' : ["friedman_mse", "mae"],    
    'gbc__subsample' : [0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0],    
    'gbc__n_estimators' : [10,50,100,150,200]},    
    
    'abc':{'abc__n_estimators' : range(10,1000,10),
    'abc__learning_rate' : [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]}, 
    
    'lgb':{'lgb__max_depth' : [4,6,8],
    'lgb__num_leaves' : [20,30,40],    
    'lgb__min_child_samples' : [18,19,20,21,22],    
    'lgb__min_child_weight' : [0.001,0.002],  
    'lgb__feature_fraction' : [0.6, 0.8, 1],
    'lgb__bagging_fraction' : [0.8,0.9,1],
    'lgb__bagging_freq' : [2,3,4],    
    'lgb__cat_smooth' : [0,10,20],    
    'lgb__n_estimators' : [10,50,100,150,200],
    'lgb__force_col_wise' : [True]},
    
    'xgb':{'xgb__n_estimators' : [50,100,150,200,300],
    'xgb__eta' : [0.05, 0.1, 0,2, 0.3],    
    'xgb__max_depth' : [3,4,5,6,7],    
    'xgb__colsample_bytree' : [0.4,0.6,0.8,1],  
    'xgb__min_child_weight' : [1,2,3,4]}
}

# %%
x_train, y_train, x_val, v_val = data_loading("your path")

model_pipeline = Pipeline(pipe_steps)

random_search_pipeline = RandomizedSearchCV(model_pipeline, param_distributions=param_grid,
                                       n_iter=1000, verbose=1, cv=10, n_jobs=-1, scoring='roc_auc')  

random_search_pipeline.fit(x_train, y_train)

y_val_proba = random_search_pipeline.predict_proba(x_val)[:, 1]
auc_val = roc_auc_score(v_val, y_val_proba)
print(auc_val)


