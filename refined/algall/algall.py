import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 绘图
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score # 数据划分，交叉验证
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc

dataset = pd.read_csv('./refined/onedonesample.csv')

# dataset.head()

# corr_var = dataset.corr()
# corr_var

# # 使用柱状图的方式画出标签个数统计
# p = dataset.ards_label.value_counts().plot(kind="bar", color=["brown", "orange"])
# plt.savefig('./refined/result/barplot.png')

# p=sns.pairplot(dataset, hue = 'ards_label')
# plt.savefig('./refined/result/pairplot.png')

# plt.figure(figsize=(10, 7.5))
# sns.heatmap(corr_var, annot=True, cmap='BuPu')
# plt.savefig('./refined/result/heatmap.png')

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

# lgbm
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
# 训练模型
lgbm_classifier.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = lgbm_classifier.predict(X_test)

# rf
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# 训练模型
rf_classifier.fit(X_train, y_train)
# 在测试集上进行预测
y_pred_rf = rf_classifier.predict(X_test)

# 计算RF的ROC曲线
y_score_rf = rf_classifier.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# 计算LGBM的ROC曲线
y_score_lgbm = lgbm_classifier.predict_proba(X_test)[:, 1]
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_score_lgbm)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

plt.figure()
lw = 2
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=lw, label='RF ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot(fpr_lgbm, tpr_lgbm, color='green', lw=lw, label='LGBM ROC curve (area = %0.2f)' % roc_auc_lgbm)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('./refined/algall/roc.png')

