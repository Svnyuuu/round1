import pandas as pd
from sklearn.utils import shuffle

# 读取
df_qall = pd.read_csv('./data/qall.csv')
df_qb = pd.read_csv('./data/qb_selected.csv')

# 两个表连起来
df_combined = pd.concat([df_qall, df_qb])

# 删第一列（subject id）
df_combined = df_combined.iloc[:, 1:]

# 缺失值用mean填充
df_combined = df_combined.fillna(df_combined.mean())

# 随机打乱行
df_combined = shuffle(df_combined)

# 存
df_combined.to_csv('./processed/combined.csv', index=False)
