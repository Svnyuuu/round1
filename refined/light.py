import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 绘图
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score # 数据划分，交叉验证
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc

dataset = pd.read_csv('./refined/onedone.csv')

dataset.head()

corr_var = dataset.corr()
corr_var

# 使用柱状图的方式画出标签个数统计
p = dataset.ards_label.value_counts().plot(kind="bar", color=["brown", "orange"])
plt.savefig('./refined/result/barplot.png')

p=sns.pairplot(dataset, hue = 'ards_label')
plt.savefig('./refined/result/pairplot.png')

plt.figure(figsize=(10, 7.5))
sns.heatmap(corr_var, annot=True, cmap='BuPu')
plt.savefig('./refined/result/heatmap.png')

# 数据提取
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print('X_train:', np.shape(X_train))
print('y_train:', np.shape(y_train))
print('X_test:', np.shape(X_test))
print('y_test:', np.shape(y_test))

# 数据标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# RF
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
# 训练模型
lgbm_classifier.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = lgbm_classifier.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
with open('./refined/result/output.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.3f}\n')
    f.write('RF:\n')
    f.write(classification_report(y_test, y_pred))
    f.write(f'roc auc score: {roc_auc_score(y_test, y_pred):.3f}\n')

# 计算ROC曲线
y_score = lgbm_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('./refined/result/roc.png')

