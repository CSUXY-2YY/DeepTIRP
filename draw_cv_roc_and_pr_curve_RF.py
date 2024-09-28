# %%
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RandomizedSearchCV

def draw_cv_roc_and_pr_curve_RF(cv, X, y, fileroc):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    """
    # Creating ROC Curve with Cross Validation

    aucs = []
    
    y_real = []
    y_proba = []
    
    #Save model and weights to file
    # es = EarlyStopping(monitor='val_roc_auc', mode='max', verbose=1, patience=15)

    font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 16}
    figsize=6.2, 6.2
    
    ########ROC_figure
    figure1, ax1 = plt.subplots(figsize=figsize)
    ax1.tick_params(labelsize=18)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]  

    ########ROC_figure
    figure2, ax2 = plt.subplots(figsize=figsize)
    ax2.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels] 

    i = 0
    for train, test in cv.split(X, y):
        pipe_steps = [('rb', RandomForestClassifier())] ## Using RandomForest as a example
        model_pipeline = Pipeline(pipe_steps)
        param_grid = {    
            'rb__max_depth' : range(5,200,5),
            'rb__min_samples_split': range(5,200,5),
            'rb__min_samples_leaf' : range(5,200,5),
            'rb__bootstrap' : [True],
            'rb__criterion' : ["gini", "entropy"],
            'rb__n_estimators' : [50,100,150,200,250,300]}

        random_search_pipeline = RandomizedSearchCV(model_pipeline, param_distributions=param_grid,
                                            n_iter=500, verbose=1, cv=5, n_jobs=-1, scoring='roc_auc')  
        random_search_pipeline.fit(X[train], y[train])
        probas_ = random_search_pipeline.predict_proba(X[test])[:, 1]
    
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        roc_auc_value = auc(fpr, tpr)
        aucs.append(roc_auc_value)
        ax1.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc_value))
        
        # Plotting each individual PR Curve
        precision, recall, _ = precision_recall_curve(y[test], probas_)
        ax2.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.4f)' % (i, average_precision_score(y[test], probas_)))

        y_real.append(y[test])
        y_proba.append(probas_)

        del random_search_pipeline
        
        i += 1
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)    
    
    lable_probas=np.c_[y_real, y_proba]                
    with open(fileroc + '/Evalution_lable_probas.txt',"w+") as f:
        for j in range(len(lable_probas)):           
            f.write(str(lable_probas[j][0]) + '\t')
            f.write(str(lable_probas[j][1]) + '\n')

    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey',
             label='Luck', alpha=.8)

    mean_fpr, mean_tpr, mean_thresholds = roc_curve(y_real, y_proba)
    mean_auc_value = auc(mean_fpr, mean_tpr)
    ax1.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f)' % (mean_auc_value),
             lw=2, alpha=.8)
    
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate', font1)
    ax1.set_ylabel('True Positive Rate', font1)
    title1 = 'Cross Validated ROC Curve'
    ax1.set_title(title1, font1)
    ax1.legend(loc="lower right")
    figure1.savefig(fileroc + '/' + 'All_feature_CV5_roc.png', dpi=300, bbox_inches = 'tight')
    
    ##############PR_Curve

    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ax2.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.4f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Recall', font1)
    ax2.set_ylabel('Precision', font1)
    title2 = 'Cross Validated PR Curve'
    ax2.set_title(title2, font1)
    ax2.legend(loc="lower left")
    figure2.savefig(fileroc + '/' + 'All_feature_CV5_PR.png', dpi=300, bbox_inches = 'tight')  


