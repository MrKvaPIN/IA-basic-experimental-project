import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
                                                                                                                                                                                                      # `读取Excel文件`
df = pd.read_excel('CW_Data.xlsx')
# 初始化变量定义
Total = df['Total']
MCQ = df['MCQ']
programme = df['Programme']
test_columns1 = ['Gender','Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']
test_columns2 = ['Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']
test_columns3 = ['Grade', 'Total','MCQ']
allcolumns = ['Index','Gender','Programme','Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']
columns = [ 'Gender','Grade','Total','MCQ','Q1','Q2','Q3','Q4','Q5']


'''
数据预处理
'''
# 数据标准化
scaler = MinMaxScaler()
# df_scaled_box = pd.DataFrame(scaler.fit_transform(df[allcolumns]), columns=allcolumns)
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
# 将标准化后的数据保存为CSV文件
df_scaled.to_csv('scaled_data.csv', index=False)
# 读取csv文件
df_scaled = pd.read_csv('scaled_data.csv')
df_scaled['Programme'] = programme
# 准备数据集，X为特征，y为标签
X = df_scaled[test_columns3]
y = df_scaled['Programme']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=88)


'''
功能函数
'''

# 生成混淆矩阵
def confusion_matrix_plot(y_test, y_pred,model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 可视化混淆矩阵的热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix of {}'.format(model_name))
    plt.show()
# Lasso特征子集选择
def LassoFeatureSelection():
    # 创建Lasso模型
    lasso_model = Lasso(alpha=0.05)  # 调整alpha参数以控制正则化程度

    # 拟合Lasso模型
    lasso_model.fit(X, y)

    # 获取选择的特征
    selected_features = [i for i, coef in enumerate(lasso_model.coef_) if coef != 0]

    print("Selected features indices:", selected_features)


'''
# 模型函数阵列
'''
def decision_trees():
    # 创建决策树分类器模型
    clf = DecisionTreeClassifier()

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    confusion_matrix_plot(y_test, y_pred,'decision_tree')
def decision_tree():
    # 创建决策树分类器模型
    clf = DecisionTreeClassifier()
    # 定义参数网格，自动分类调整
    param_grid = {
        'criterion': ['gini', 'entropy'],#控制决策树划分的标准
        'max_depth': np.arange(6, 10, 1), #控制决策树深度
        'min_samples_leaf': np.arange(1, 15, 1),#控制决策树叶节点的样本数量
        'min_samples_split': np.arange(2, 10, 1) #影响决策树内部节点的拆分过程
    # 最佳参数组合:  {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 1, 'min_samples_split': 2}
    }
    # 使用GridSearchCV进行参数搜索，同时进行3折的交叉验证
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    # 输出最佳参数组合
    print("最佳参数组合: ", grid_search.best_params_)

    # 使用最佳参数训练模型
    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = best_clf.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 计算F1分数
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"准确率：{accuracy}")
    print(f"F1 score for selected feature subset: {f1}")
    # 生成混淆矩阵
    confusion_matrix_plot(y_test, y_pred,'decision_tree')
def random_forest():
    # 创建随机森林分类器模型
    rf_clf = RandomForestClassifier()

    # 使用GridSearchCV进行参数搜索
    rf_param_grid = {
        'n_estimators': [18],  # 决策树数量
        'max_depth': [3],# 决策树深度
        'min_samples_split': [13],# 决策树内部节点再划分所需最小样本数
        'min_samples_leaf': [3]# 决策树叶子节点最少样本数
    # 随机森林最佳参数组合:  {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 13, 'n_estimators': 18}
    }

    rf_grid_search = GridSearchCV(rf_clf, rf_param_grid, cv=3)
    rf_grid_search.fit(X_train, y_train)

    # 输出最佳参数组合
    print("随机森林最佳参数组合: ", rf_grid_search.best_params_)

    # 使用最佳参数训练随机森林模型
    best_rf_clf = rf_grid_search.best_estimator_
    best_rf_clf.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred_rf = best_rf_clf.predict(X_test)
    # 计算准确率
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林分类器准确率：{accuracy_rf}")
    # 生成混淆矩阵
    confusion_matrix_plot(y_test, y_pred_rf,'random_forest')
def svm():
    # 创建SVM模型
    svm_model = SVC(kernel='linear')
    param_grid = {
        'C': np.arange(1, 10, 1),  # 惩罚参数范围从1到100，步长为1
        'kernel': ['rbf'],  # 选择线性核、高斯核和多项式核
        'gamma': ['scale', 'auto'],  # 选择gamma值为'scale'或者'auto'
        'coef0': np.arange(0.5, 1.5, 0.1),  # 核函数中的独立项选择范围
        # Best Parameters: {'C': 1, 'coef0': 0.8999999999999999, 'gamma': 'scale', 'kernel': 'poly'}
        # Best Model: SVC(C=1, coef0=0.8999999999999999, kernel='poly')
    }
    grid_search = GridSearchCV(SVC(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f'Best Parameters: {best_params}')
    print(f'Best Model: {best_model}')
    # 进行预测并计算准确率
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM Accuracy: {accuracy}')
    # 生成混淆矩阵
    confusion_matrix_plot(y_test, y_pred,'SVM')
def naive_bayes():
    # 创建朴素贝叶斯模型
    model = GaussianNB()
    # 训练模型
    model.fit(X_train, y_train)
    # 进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"f准确率：{f1}")
    print(f'Naive Bayes {model} Accuracy: {accuracy}')
    # 生成混淆矩阵
    confusion_matrix_plot(y_test, y_pred,'naive_bayes {}'.format(model))


'''
集成分类算法模型
'''
def adaboostclassifier():
    # 实例化不同的弱分类器
    dt_clf1 = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=1, min_samples_split=2)
    dt_clf2 = RandomForestClassifier(max_depth=6, min_samples_leaf=2, min_samples_split=7, n_estimators=20)
    dt_clf3 = SVC(C=5, gamma='scale',kernel='rbf', probability=True,coef0=0.8999999999999999)
    dt_clf4 = GaussianNB()

    # 创建AdaBoost模型并传入多个弱分类器,调整学习率使准确率提高
    adaboost_model = AdaBoostClassifier(n_estimators=4,algorithm='SAMME',learning_rate=0.75)

    # 添加弱分类器到AdaBoost模
    adaboost_model.estimators_ = [dt_clf1, dt_clf2, dt_clf3,dt_clf4]
    adaboost_model.fit(X_train, y_train)
    # 使用模型进行预测
    y_pred = adaboost_model.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"AdaBoost Model Accuracy: {accuracy}")
    # 生成混淆矩阵
    confusion_matrix_plot(y_test, y_pred,'adaboost')


def votingclassifier():
    # 初始化四种不同的模型
    decision_tree_model = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=1, min_samples_split=2)
    random_forest_model = RandomForestClassifier(max_depth=6, min_samples_leaf=2, min_samples_split=7, n_estimators=20)
    svm_model = SVC(C=5, gamma='scale',kernel='rbf', probability=True)
    naive_bayes_model = GaussianNB()

    # 使用Voting Classifier集成四种模型
    ensemble_model = VotingClassifier(estimators=[
        ('Decision Tree', decision_tree_model),
        ('Random Forest', random_forest_model),
        ('SVM', svm_model),
        ('Naive Bayes', naive_bayes_model)
    ], voting='soft')

    # 在训练集上拟合集成模型
    ensemble_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = ensemble_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble Model Accuracy: {accuracy}")
    confusion_matrix_plot(y_test, y_pred,'voting_classifier')


'''
# main function
'''
# LassoFeatureSelection()
# decision_trees()
decision_tree()
# random_forest()
# svm()
# naive_bayes()
# adaboostclassifier()
# votingclassifier()


