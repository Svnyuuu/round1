import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle

# 假设我们知道有 5 个插补数据集
for i in range(5):
    file_path = f'imputation/result/imputed_dataset_{i}.csv'
    dataset = pd.read_csv(file_path)

    # 数据提取
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # 处理缺失值并转换数据类型为 float64
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = X.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 训练 DecisionTreeClassifier 模型
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = clf.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # 计算 ROC 曲线
    y_score = clf.predict_proba(X_test)[:, 1] #两列只取最后一列？
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # # 保存 ROC 曲线
    # plt.savefig(f'roc_curve_{i}.png')

    # 保存 ROC 曲线和 AUC 分数
    roc_auc_score_path = f'imputation/result/roc_auc_{i}.txt'
    with open(roc_auc_score_path, 'w') as file:
        file.write(f'ROC AUC Score: {roc_auc}\n')

    # 保存模型
    model_path = f'imputation/result/model_{i}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

    # 保存预测结果
    predictions_path = f'imputation/result/predictions_{i}.csv'
    y_pred_df = pd.DataFrame(y_pred, columns=['Prediction'])
    y_pred_df.to_csv(predictions_path, index=False)

    # 保存准确率
    accuracy_path = f'imputation/result/accuracy_{i}.txt'
    with open(accuracy_path, 'w') as file:
        file.write(f'Accuracy: {accuracy:.2f}\n')
