from utils import *
from sklearn.model_selection import train_test_split

train_file_path = "../data/processed/cleaned_train.csv"
train = load_data(train_file_path)

# Drop the ID column
train = train.drop("ID", axis=1)
train = train.reset_index(drop=True)

# Separate the target column
train_target = train["Segmentation"]

# Separate categorical and numerical columns
cat_cols, num_cols = separate_categorical_numerical(train)

# Normalize numerical values by minmax scaling because the distributions
# are skewed
train_num = normalize_dataframe(train[num_cols], train=True, scaler='minmax')

# Encode categorical values based on type of encoding
onehot_cols = ['Ever_Married', 'Gender', 'Graduated', 'Var_1']
ordinal_cols = ['Spending_Score']
binary_cols = ['Profession']

train_onehot = encode_df(train[onehot_cols], 'onehot', train=True)
train_ordinal = encode_df(train[ordinal_cols], 'ordinal', train=True)
train_binary = encode_df(train[binary_cols], 'binary', train=True)

# Concatenate the encoded dataframes with the normalized and scaled dataframe
train_processed = pd.concat([train_num, train_ordinal, train_binary, train_onehot], axis=1).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(train_processed, train_target, test_size = 0.2, random_state = 0)

results = train_classifier(X_train, X_test, y_train, y_test, compare=True)

print(results)

