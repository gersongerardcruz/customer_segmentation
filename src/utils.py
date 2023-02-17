import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def load_data(file_path, index_col=0, file_type='csv'):
    """
    Loads data into a pandas DataFrame.

    Parameters:
    file_path (str): The file path to the data.
    index_col (int, optional): The index to be used. Defaults to 0. 
    file_type (str): The type of file to be loaded. 
                     Default is 'csv'. 
                     Other options include 'excel', 'json', 'hdf', 'parquet', etc.

    Returns:
    pandas DataFrame: The loaded data.

    """
    # Load the data into a pandas DataFrame
    if file_type == 'csv':
        df = pd.read_csv(file_path, index_col=index_col)
    elif file_type == 'excel':
        df = pd.read_excel(file_path, index_col=index_col)
    elif file_type == 'json':
        df = pd.read_json(file_path, index_col=index_col)
    elif file_type == 'hdf':
        df = pd.read_hdf(file_path, index_col=index_col)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path, index_col=index_col)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return df


def compute_basic_stats(df):
    """
    This function takes a pandas dataframe column and returns the following statistics:
    1. Total missing values
    2. Percentage of missing values with respect to the entire dataframe
    3. Unique values present in the column
    4. Basic statistics (mean, median, mode, standard deviation) of the column
    5. 25th and 75th percentile of the column
    
    The function will handle both numerical and categorical columns, providing different statistics for each type.
    
    Parameters:
    col (pandas series): a single column from a pandas dataframe
    
    Returns:
    dict: a dictionary containing the statistics for the input column
    """
 
    stats = {}
    for col in df.columns:
        # If the column is numerical, calculate and return the statistics
        if df[col].dtype in ['float64', 'int64']:
            missing_values = df[col].isnull().sum()
            percent_missing = missing_values / len(df) * 100
            uniques = df[col].nunique()
            mean = df[col].mean()
            median = df[col].median()
            mode = df[col].mode().values[0]
            min = df[col].min()
            percentiles = df[col].quantile([0.25, 0.75])
            max = df[col].max()
            std = df[col].std()
            stats[col]={
                'missing_values': missing_values,
                'percent_missing': percent_missing,
                'uniques': uniques,
                'mean': mean,
                'median': median,
                'mode': mode,
                'min': min,
                '25th_percentile': percentiles[0.25],
                '75th_percentile': percentiles[0.75],
                'max': max,
                'std': std,
            }
        
        # If the column is categorical, calculate and return the mode
        else:
            missing_values = df[col].isnull().sum()
            percent_missing = missing_values / len(df) * 100
            uniques = df[col].nunique()
            mode = df[col].mode().values[0]
            stats[col]={
                'missing_values': missing_values,
                'percent_missing': percent_missing,
                'uniques': uniques,
                'mode': mode
            }
    
    return pd.DataFrame(stats)


def inspect_column(df, col_name):
    """
    Inspect a single column in a pandas DataFrame.
    
    Parameters:
    df (pandas DataFrame): The DataFrame to inspect.
    col_name (str): The name of the column to inspect.
    
    Returns:
    None
    """
    # Calculate the number of missing values
    missing_values = df[col_name].isna().sum()
    print(f"Number of missing values in {col_name}: {missing_values}")
    
    # Print the unique values in the column
    unique_values = df[col_name].nunique()
    print(f"Number of unique values in {col_name}: {unique_values}")
    print(f"Unique values in {col_name}: {df[col_name].unique()}")
    
    # Print basic statistics of the column
    print(f"Basic statistics of {col_name}:")
    print(df[col_name].describe())
    
    # Check if the column is quantitative or not
    if df[col_name].dtype in [np.int64, np.float64]:
        # Check for outliers using the IQR method
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]
        print(f"Number of outliers in {col_name}: {len(outliers)}")
        print(f"Outliers in {col_name}: {outliers}")
    else:
        # Print the mode of the column
        mode = df[col_name].mode().values[0]
        print(f"Mode of {col_name}: {mode}")

def get_correlation(df, col_name):
    """
    This function takes a dataframe and a column name, and returns the correlation of that column with all other columns in the dataframe.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    col_name (str): The name of the column to compute correlations for.
    
    Returns:
    pandas.Series: A series with the correlation values of the input column with all other columns in the dataframe.
    """
    
    # Calculate the correlation between the input column and all other columns
    corr = df.corr()[col_name]
    
    return corr


def knn_imputer(df, columns):
    """
    Fill in the missing values in a pandas dataframe using KNN imputation.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to fill missing values in
    columns (list): The columns in the dataframe to use for KNN imputation
    
    Returns:
    pandas.DataFrame: The dataframe with filled in missing values
    pandas.DataFrame: The imputed dataframe for reference
    """
    # Subset the dataframe to only include the desired columns
    df_subset = df[columns].copy()
    
    # Fit the KNNImputer on the subsetted dataframe
    imputer = KNNImputer()
    df_imputed = imputer.fit_transform(df_subset)
    
    # Convert the imputed array back to a dataframe
    df_imputed = pd.DataFrame(df_imputed, columns=columns)
    
    # Merge the imputed dataframe back into the original dataframe
    df_imputed.index = df.index
    df_filled = df.drop(columns, axis=1).merge(df_imputed, left_index=True, right_index=True)
    
    return df_filled, df_imputed


def separate_categorical_numerical(df):
    """
    Separate the categorical and numerical columns of a pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    
    Returns:
    tuple: A tuple containing two pandas DataFrames, one for the categorical columns and one for the numerical columns.
    """
    # Select the columns with data type 'object', which are assumed to be categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Select the columns with data types other than 'object', which are assumed to be numerical
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    # Return two separate DataFrames, one for the categorical columns and one for the numerical columns

    return categorical_cols, numerical_cols


def create_contingency_tables(df):
    """
    Create contingency tables for selected categorical columns in a pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame with categorical columns.
    
    Returns:
    dict: A dictionary where the keys are the names of the categorical columns and the values are the contingency tables.
    """

    contingency_tables = {}
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                ct = pd.crosstab(df[col1], df[col2])
                contingency_tables[col1 + ' x ' + col2] = ct
    
    contingency_tables_df = pd.concat(contingency_tables, axis=1)

    return contingency_tables_df


def plot_categorical_comparison(df, column):
    """
    This function plots bar charts over subplots that compares the values of a column
    with all the other categorical columns in a dataframe. The subplots are automatically 
    adjusted to the number of columns and a maximum of two graphs per row is set.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe
    column (str): The column to be compared with the other categorical columns
    
    Returns:
    None
    """
    
    # Get the categorical columns from the dataframe
    categorical_columns = df.select_dtypes(include=['category', object]).columns
    # Remove the column to be compared
    categorical_columns = categorical_columns.drop(column)
    
    # Calculate the number of rows needed to plot the subplots
    nrows = math.ceil(len(categorical_columns) / 2)
    # Calculate the number of columns needed to plot the subplots
    ncols = min(len(categorical_columns), 2)
    
    # Initialize the subplot
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*5))
    
    # Flatten the subplot axis array
    ax = np.array(ax).flatten()
    
    # Iterate over the categorical columns
    for i, cat_col in enumerate(categorical_columns):
        # Plot the bar chart for the comparison
        df.groupby([column, cat_col])[cat_col].count().unstack().plot(kind='bar', stacked=True, ax=ax[i])
        # Set the title for the subplot
        ax[i].set_title(f"{column} vs {cat_col}")
        ax[i].legend(loc="upper left", fontsize="xx-small")
        
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_categorical_numerical_comparison(df, column):
    """
    Plot bar charts over subplots that compare one column to all other numerical columns in a DataFrame.
    The point of comparison is the mean of each numerical column.
    
    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame
    column: str
        The categorical column to be compared to the numerical columns
    
    Returns
    -------
    None
    """
    
    numerical_columns = [col for col in df.columns if df[col].dtype in [np.number]]

    # Calculate the number of rows needed to plot the subplots
    nrows = math.ceil(len(numerical_columns) / 2)
    # Calculate the number of columns needed to plot the subplots
    ncols = min(len(numerical_columns), 2)

    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 10 * nrows))
    ax = ax.flatten()
    
    for i, num_col in enumerate(numerical_columns):
        axi = ax[i]
        axi.set_title(num_col)
        for category in sorted(df[column].unique()):
            data = df[df[column] == category][num_col]
            mean = data.mean()
            axi.bar(category, mean, label=category)
        axi.legend(loc="best")
    
    plt.tight_layout()
    plt.show()


def plot_distributions(df, segment_col, figsize=(20, 10)):
    """
    This function plots the distributions of all numerical columns in a dataframe grouped by the segment column.
    It creates subplots with two graphs per row.
    
    Parameters:
        df (pandas dataframe): The dataframe to plot.
        segment_col (str): The name of the column to group the distributions by.
        figsize (tuple, optional): The size of the plot. Default is (20, 10).
    
    Returns:
        None
    """

    numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    
    for i, num_col in enumerate(numerical_cols):
        print(f"{num_col}")
        df[num_col].hist(by=df[segment_col])
        plt.tight_layout()
        plt.show()


def normalize_dataframe(df, train_df, train: bool, save_scaler: bool, scaler='standard'):
    """
    Normalize all columns in a pandas dataframe using either StandardScaler or MinMaxScaler.
    
    Parameters:
    df (pandas dataframe): The data to be normalized.
    train_df (pandas dataframe): The training data for fitting the scaler.
    train (bool): Whether the data is from the training set or not. If True, the data is from the training set.
    save_scaler (bool): Whether to save the scaler parameters. 
    scaler (str, optional): The type of scaler to use. Must be either 'standard' or 'minmax'. Default is 'standard'.
    
    Returns:
    pandas.DataFrame, pandas.DataFrame: The normalized training data and test data as dataframes.
    """

    if scaler == 'standard':
        # Use StandardScaler to normalize the data
        scaler = StandardScaler()
    elif scaler == 'minmax':
        # Use MinMaxScaler to normalize the data
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaler type: {scaler}. Must be either 'standard' or 'minmax'.")
    
    # Fit the scaler to the training data
    scaler.fit(train_df)

    # Save the scaler parameters to file
    if save_scaler == True:
        if scaler == 'standard':
            np.save('standard_scaler_params.npy', [scaler.mean_, scaler.var_])
        else:
            np.save('minmax_scaler_params.npy', [scaler.min_, scaler.scale_])

    if train: 
        # Transform the training data
        normalized_train_data = scaler.transform(df)
        
        # Return the normalized data as a dataframe
        normalized_df_train = pd.DataFrame(normalized_train_data, columns=df.columns)

        return normalized_df_train
    
    else:
        # Transform the test data using the same parameters
        normalized_test_data = scaler.transform(df)
        
        # Return the normalized test data as a dataframe
        normalized_df_test = pd.DataFrame(normalized_test_data, columns=df.columns)

        return normalized_df_test


def encode_df(df, encoder_type, train: bool):
    """
    Perform encoding on a Pandas dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to be encoded.
    encoder_type (str): The type of encoding to perform. Must be one of 'onehot', 'ordinal', or 'label'.
    train (bool): Whether the data is from the training set or not. If True, the data is from the training set.
    
    Returns:
    pandas.DataFrame: The encoded dataframe.
    dict: A dictionary containing a mapping of the original value and its encoded value for each column. Values encoded
    as strings for better json interaction. 
    """

    columns = df.columns
    print(columns)
    df_encoded = df.copy()
    
    encoders = {}
    mappings = {}

    for column in columns:
            if encoder_type == 'onehot':
                encoder = OneHotEncoder(handle_unknown='ignore')
                if train:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]).toarray(), columns=encoder.get_feature_names_out([column]))], axis=1)
                else:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]).toarray(), columns=encoder.get_feature_names_out([column]))], axis=1)
                df_encoded = df_encoded.drop(column, axis=1)
                mappings[column] = dict(zip(range(len(encoder.categories_[0])), encoder.categories_[0]))

            elif encoder_type == 'ordinal':
                encoder = OrdinalEncoder()
                if train:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[[column]])
                else:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[[column]])

                # Encode the data and save the mapping to a dictionary
                mappings[column] = {label: str(idx) for idx, label in enumerate(encoder.categories_[0])}

            elif encoder_type == 'label':
                encoder = LabelEncoder()
                if train:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[column])
                else:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[column])
                mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                mappings[column] = {k: str(v) for k, v in mappings[column].items()}

            elif encoder_type == 'binary':
                encoder = ce.BinaryEncoder()
                if train:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]), columns=encoder.get_feature_names_out())], axis=1)
                else:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]), columns=encoder.get_feature_names_out())], axis=1)
                df_encoded = df_encoded.drop(column, axis=1)
                mappings[column] = dict(zip(range(len(encoder.get_feature_names())), encoder.get_feature_names()))

            else:
                raise ValueError("Encoder type must be one of 'onehot', 'ordinal', 'label', or 'binary'.")
    
    return df_encoded, mappings


def train_classifier(X_train, X_test, y_train, y_test, classifier=None, compare=False):
    """
    Train a classifier on the input data and target.

    Parameters:
    X_train (pandas dataframe): Train data
    X_test (pandas dataframe): Test data
    y_train (pandas series): Train target
    y_test (pandas series): Test target
    classifier (str, optional): Classifier to use. Can be 'logistic', 'naive bayes', 'knn', 'decision tree', or 'random forest'. 
                                 If not specified, all classifiers will be trained and compared.
    compare (bool, optional): If True, all classifiers will be trained and compared. Default is False.

    Returns:
    classifier (sklearn classifier object): Trained classifier object.
    accuracy (float): Accuracy of the trained classifier.
    classifier name (str): Name of the best performing classifier. 
    """

    
    # Dictionary of classifiers
    classifiers = {
        'logistic': LogisticRegression(),
        'naive bayes': GaussianNB(),
        'knn': KNeighborsClassifier(),
        'decision tree': DecisionTreeClassifier(),
        'random forest': RandomForestClassifier(), 
        'gradient boost': GradientBoostingClassifier()
    }
    
    # Train the specified classifier or all classifiers and compare their results
    if classifier is not None:
        # Train the specified classifier
        model = classifiers[classifier]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    elif compare:
        # Train all classifiers and compare their results
        results = []
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((name, accuracy))
        
        # Sort results in descending order of accuracy
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Print the results
        print("Classifier comparison:")
        for name, accuracy in results:
            print(f"{name}: {accuracy}")

        model = classifiers[results[0][0]] 
        return model, results[0][0]
 

def hyperparameter_tuning(model, X_train, y_train, params, search_type='grid', n_iter=10, cv=5, random_state=0):
    """
    Performs hyperparameter tuning on the input model using GridSearchCV or RandomizedSearchCV.
    
    Parameters:
    model (Estimator): An instance of a scikit-learn estimator that implements fit and predict methods.
    X_train (ndarray): The training set.
    y_train (ndarray): The target values of the training set.
    params (dict): The parameters to use in the search.
    search_type (str): The type of search to perform. Can be 'grid' or 'random'.
    n_iter (int): The number of iterations to perform in RandomizedSearchCV. Only used if search_type is 'random'.
    cv (int): The number of cross-validation folds to use in the search.
    random_state (int): The random state to use for the search.
    
    Returns:
    The best estimator found from the search.
    """
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1)
    elif search_type == 'random':
        search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=random_state)
    else:
        raise ValueError('Invalid search_type: {}'.format(search_type))
    
    search.fit(X_train, y_train)
    
    print('Best parameters:', search.best_params_)
    print('Best score:', search.best_score_)
    
    return search.best_estimator_
