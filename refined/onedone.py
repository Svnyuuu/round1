import pandas as pd
from sklearn.utils import shuffle

# 读取
df_qall = pd.read_csv('./data/qall.csv')
df_qb = pd.read_csv('./data/qb_selected.csv')

# 随机打乱qb
df_qb = shuffle(df_qb)
# 删除qb前184行
df_qb = df_qb.iloc[184:]

# 两个表连起来
data = pd.concat([df_qall, df_qb])

# 删第一列（subject id）
data = data.iloc[:, 1:]

# 缺失值用mean填充
data = data.fillna(data.mean())

# 随机打乱行
data = shuffle(data)

# ards_count = data['ards_label'].value_counts()
# print(ards_count)
data.to_csv('./refined/onedone.csv', index=False)