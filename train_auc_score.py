# %%
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier #K近邻算法
from sklearn.ensemble import GradientBoostingClassifier #梯度提升树
from xgboost import XGBClassifier #极度梯度提升树
from sklearn.ensemble import AdaBoostClassifier #AdaBoost
import lightgbm as lgb #LGB
from sklearn.metrics import roc_auc_score #auc

# %%
def train_and_score(X_train,y_train,X_test,y_test):
    rs = 202404
    models = (
        LogisticRegression(random_state=rs), #逻辑回归 LR
        DecisionTreeClassifier(random_state=rs), #决策树 DT
        RandomForestClassifier(random_state=rs), #随机森林 RF
        SVC(probability=True, random_state=rs), #支持向量机 SVM
        KNeighborsClassifier(), #K近邻算法 KNN
        GradientBoostingClassifier(random_state=rs), # 梯度提升树 GBDT
        XGBClassifier(random_state=rs),#极度梯度提升树 XGBoost
        AdaBoostClassifier(random_state=rs), #集成学习分类器
        lgb.LGBMClassifier(random_state=rs) #lightgbm
    )
    res = []
    for model in models:
        model_e = model.fit(X_train,y_train)
        y_test_proba = model_e.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_test_proba)
        res.append([model, auc])
    return res


