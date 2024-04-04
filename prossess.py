import pandas as pd
from sklearn.utils import shuffle

# Read the CSV files
df_qall = pd.read_csv('./data/qall.csv')
df_qb = pd.read_csv('./data/qb_selected.csv')

# Combine the dataframes
df_combined = pd.concat([df_qall, df_qb])

df_combined = df_combined.iloc[:, 1:]

# Fill missing values with mean
# df_combined = df_combined.fillna(df_combined.mean())
# Shuffle the rows of the combined dataframe
df_combined = shuffle(df_combined)
# Save the shuffled dataframe to a new CSV file
df_combined.to_csv('./processed/combined.csv', index=False)
