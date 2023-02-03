from utils import *

file_path = "../data/processed/cleaned_train.csv"
df = load_data(file_path)

# Drop the ID column
df = df.drop("ID", axis=1)
df = df.reset_index(drop=True)

# Separate categorical and numerical columns
cat_cols, num_cols = separate_categorical_numerical(df)

# Normalize numerical values by minmax scaling because the distributions
# are skewed
df_num = normalize_dataframe(df[num_cols], train=True, scaler='minmax')

# Encode categorical values based on type
df_cat = encode_df(df[cat_cols], 'onehot', train=True)

df_processed = pd.concat([df_num, df_cat], axis=1).reset_index(drop=True)

print(df_processed)