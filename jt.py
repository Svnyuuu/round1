import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 绘图
from sklearn.linear_model import LogisticRegression # 逻辑回归

from sklearn.model_selection import train_test_split, cross_val_score # 数据划分，交叉验证
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc

dataset = pd.read_csv('./processed/combined.csv')
dataset.describe()

# 数据相关性
corr_var = dataset.corr()
corr_var

dataset.head() # 默认前五行

# 使用柱状图的方式画出标签个数统计
p = dataset.ards_label.value_counts().plot(kind="bar", color=["brown", "orange"])

# 可视化数据分布
p=sns.pairplot(dataset, hue = 'ards_label')

plt.figure(figsize=(10, 7.5))
sns.heatmap(corr_var, annot=True, cmap='BuPu') # 热力图 浅蓝色（Blue）到深紫色（Purple）的渐变

# 数据提取
X = dataset.iloc[:, :-1].values # 除了label
y = dataset.iloc[:, -1].values # label

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40) # 随机种子
print('X_train:', np.shape(X_train))
print('y_train:', np.shape(y_train))
print('X_test:', np.shape(X_test))
print('y_test:', np.shape(y_test))

# 数据标准化
sc = StandardScaler() # 标准正态分布
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LR训练
classifier_LR = LogisticRegression(tol=1e-10,solver="lbfgs",max_iter=10000)
classifier_LR.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = classifier_LR.predict(X_test)
# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('LR:\n', classification_report(y_test, y_pred))

# 计算混淆矩阵:TP FP FN TN
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

# 计算ROC面积
print(roc_auc_score(y_test, y_pred))

# 画出ROC曲线
# 预测概率 ，给定测试数据，返回预测的每个类别的概率。
y_score = classifier_LR.predict_proba(X_test)[:, 1]
# 第一列是观察值属于负类的概率，第二列是观察值属于正类的概率
# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_score) # fpr:假正例率 tpr:真正例率
roc_auc = auc(fpr, tpr) # 计算AUC
plt.figure() # 创建图
lw = 2 # 线宽
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0]) # x轴范围
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc="lower right") #图例
plt.show()

# # 设定阀值
# thresh_count = data.shape[0]*0.8
# # 若某一列数据缺失的数量超过20%就会被删除
# data = data.dropna(thresh=thresh_count, axis=1)
# p = msno.bar(data)
