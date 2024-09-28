# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict  
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, recall_score, precision_score

def calculate_metrics(gt, pred_score, folder, name):
    """
    :param gt: 数据的真实标签，一般对应二分类的整数形式，例如:y=[1,0,1,0,1]
    :param pred: 输入数据的预测值，因为计算混淆矩阵的时候，内容必须是整数，所以对于float的值，应该先调整为整数
    :return: 返回相应的评估指标的值
    """
    
    scores_two = np.array([[1-s, s] for s in pred_score])
    pred = np.argmax(scores_two,axis = 1)

    fig, ax2 = plt.subplots(figsize=(8, 6))
    confusion = confusion_matrix(gt, pred)
    
    # confusion_percent = confusion / np.sum(confusion)
    
    sum_true = np.expand_dims(np.sum(confusion, axis=1), axis=1)
    confusion_percent = confusion / sum_true
    confusion_percent = confusion_percent * 100
    print(confusion_percent)
    
    labels = (np.asarray(["{:.0f}\n ({:.2f}%)".format(string, value)
                            for string, value in zip(confusion.flatten(),
                                                    confusion_percent.flatten())])
        ).reshape(2, 2)
    
    print(confusion)  
    print(confusion.ravel())
    sns.set_style(style='white')
    axis_labels = ['Bone-tumor', 'Bone-infection'] # labels for x-axis
    
    sns.heatmap(confusion,annot=labels,fmt="",
                cmap='Blues')
    
    ax2.set_title('Confusion Matrix')
    ax2.set_xlabel('Predict lable')
    ax2.set_ylabel('Ture lable')
    ax2.set_xticklabels(axis_labels)   
    ax2.set_yticklabels(axis_labels)  
    ax2.xaxis.tick_bottom() 
    ax2.yaxis.tick_left() 

    fig.savefig(folder + '%s_sns_heatmap_confusion_matrix.jpg' % name, bbox_inches='tight')

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    print('AUC:',roc_auc_score(gt, pred_score))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))
    print('PPV:',TP / float(TP + FP))
    print('NPV:',TN / float(TN + FN))
    print('Recall:',TP / float(TP + FN))
    print('Precision:',TP / float(TP + FP))

    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    print('F1-score:',(2*P*R)/(P+R))
    print('True Positive Rate:',round(TP / float(TP + FN)))
    print('False Positive Rate:',FP / float(FP + TN))
    print('Ending!!!------------------------------------------------------')
 
    print("the result of sklearn package")
    auc = roc_auc_score(gt,pred)
    print("sklearn auc:",auc)
    accuracy = accuracy_score(gt,pred)
    print("sklearn accuracy:",accuracy)
    recal = recall_score(gt,pred)
    precision = precision_score(gt,pred)
    print("sklearn recall:{},precision:{}".format(recal,precision))
    print("sklearn F1-score:{}".format((2*recal*precision)/(recal+precision)))



