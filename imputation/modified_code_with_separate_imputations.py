import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

combined_df = pd.read_csv('./processed/combined.csv')
# combined_df = pd.read_csv('./processed/randel.csv')

# 中值填充'average_resp_rate_score' 和 'average_po2'缺失值
combined_df['average_resp_rate_score'].fillna(combined_df['average_resp_rate_score'].median(), inplace=True)
combined_df['average_po2'].fillna(combined_df['average_po2'].median(), inplace=True)

# Initialize the SimpleImputer with the median strategy
simple_imputer = SimpleImputer(strategy='median')

# Initialize KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform k-fold imputation and save each dataset
file_paths = []
for i, (train_index, test_index) in enumerate(kf.split(combined_df)):
    # Split the dataframe into training and testing sets
    train_set = combined_df.iloc[train_index]
    test_set = combined_df.iloc[test_index]

    # Fit the imputer on the training set and transform the testing set
    simple_imputer.fit(train_set[['sofa_score']])
    test_set['sofa_score'] = simple_imputer.transform(test_set[['sofa_score']])

    # Combine the training and testing sets to form the imputed dataset
    imputed_dataset = pd.concat([train_set, test_set], ignore_index=True)

    # Save the imputed dataset to a separate CSV file
    file_path = f'imputation/result/imputed_dataset_{i}.csv'
    imputed_dataset.to_csv(file_path, index=False)
    file_paths.append(file_path)

print("Imputation completed. Files saved at:", file_paths)
