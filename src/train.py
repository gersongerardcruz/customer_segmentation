from utils import *
from sklearn.model_selection import train_test_split

train_file_path = "../data/processed/cleaned_train.csv"
test_file_path = "../data/processed/cleaned_test.csv"
train = load_data(train_file_path)
test = load_data(test_file_path)

# Drop the ID column
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Separate the target column
train_target = train["Segmentation"]
test_target = test["Segmentation"]

# Separate categorical and numerical columns
cat_cols, num_cols = separate_categorical_numerical(train)

# Normalize numerical values by minmax scaling because the distributions
# are skewed
train_num = normalize_dataframe(train[num_cols], train[num_cols], train=True, scaler='minmax')
test_num = normalize_dataframe(test[num_cols], train[num_cols], train=False, scaler='minmax')

# Encode categorical values based on type of encoding
onehot_cols = ['Var_1', 'Profession']
ordinal_cols = ['Spending_Score']
label_cols = ['Gender', 'Graduated', 'Ever_Married']

train_onehot = encode_df(train[onehot_cols], 'onehot', train=True)
train_ordinal = encode_df(train[ordinal_cols], 'ordinal', train=True)
train_label = encode_df(train[label_cols], 'label', train=True)

test_onehot = encode_df(test[onehot_cols], 'onehot', train=False)
test_ordinal = encode_df(test[ordinal_cols], 'ordinal', train=False)
test_label = encode_df(test[label_cols], 'label', train=False)

# Concatenate the encoded dataframes with the normalized and scaled dataframe
train_processed = pd.concat([train_num, train_ordinal, train_onehot, train_label], axis=1).reset_index(drop=True)
test_processed = pd.concat([test_num, test_ordinal, test_onehot, test_label], axis=1).reset_index(drop=True)

print(train_processed.columns)

X_train, X_test, y_train, y_test = train_test_split(train_processed, train_target, test_size = 0.2, random_state = 0)

model, model_name = train_classifier(X_train, X_test, y_train, y_test, compare=True)

print("The best performing model is: {}".format(model_name))

# Tuning the model, list out parameters for the models we have used

logreg_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', None],
    'fit_intercept': [True, False],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 500, 1000, 2000]
}

knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

nb_params = {
    'priors': [None],
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

dt_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 5, 10]
}

rf_params = {
    'n_estimators': [10, 50, 100, 200, 300, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'bootstrap': [True, False]
}

if model_name == 'logistic':
    params = logreg_params
elif model_name == 'knn':
    params = knn_params
elif model_name == 'naive bayes':
    params = nb_params
elif model_name == 'decision tree':
    params = dt_params
else:
    params = rf_params

hyperparameter_tuning(model, X_train, y_train, params=params, search_type='grid')
