import pandas as pd
import random

data = pd.read_csv('./processed/combined.csv')
# 首先，统计ards_label为0的行数
num_zeros = (data['ards_label'] == 0).sum()

# 计算需要删除的行数
rows_to_drop = num_zeros // 2

# 随机选择并删除ards_label为0的一半行
indices_to_drop = data[data['ards_label'] == 0].index
rows_to_drop_indices = random.sample(indices_to_drop.tolist(), rows_to_drop)
data.drop(rows_to_drop_indices, inplace=True)


data.to_csv('./processed/randel.csv', index=False)