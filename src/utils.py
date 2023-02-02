import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def load_data(file_path, file_type='csv'):
    """
    Loads data into a pandas DataFrame.

    Parameters:
    file_path (str): The file path to the data.
    file_type (str): The type of file to be loaded. 
                     Default is 'csv'. 
                     Other options include 'excel', 'json', 'hdf', 'parquet', etc.

    Returns:
    pandas DataFrame: The loaded data.

    """
    # Load the data into a pandas DataFrame
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
    elif file_type == 'json':
        df = pd.read_json(file_path)
    elif file_type == 'hdf':
        df = pd.read_hdf(file_path)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path)
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