import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from env import host, username, password



# Function for acquiring and prepping my student_grades df.

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')
    
    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = grades.dropna()
    
    # Convert all columns to int64 data types.
    df = df.astype('int')
    
    return df

# Generic helper function to provide connection url for Codeup database server.

def get_db_url(db_name):
    '''
    This function uses my env file to get the url to access the Codeup database.
    It takes in a string identifying the database I want to connect to.
    '''
    return f"mysql+pymysql://{username}:{password}@{host}/{db_name}"


# Generic function that takes in a database name and a query.

def get_data_from_sql(str_db_name, query):
    '''
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    '''
    df = pd.read_sql(query, get_db_url(str_db_name))
    return df


# Mother function to acquire and prepare Telco data.

def wrangle_telco():
    """
    Queries the telco_churn database
    Returns a clean df with four columns:
    customer_id(object), monthly_charges(float), tenure(int), total_charges(float)
    """
    query = """
            SELECT 
                customer_id, 
                monthly_charges, 
                tenure, 
                total_charges
            FROM customers
            WHERE contract_type_id = 3;
            """
    df = get_data_from_sql('telco_churn', query)
    
    # Replace any tenures of 0 with 1
    df.tenure = df.tenure.replace(0, 1)
    
    # Replace the blank total_charges with the monthly_charge for tenure == 1
    df.total_charges = np.where(df.total_charges==' ', df.monthly_charges, df.total_charges) 
    
    # Convert total_charges to a float.
    df.total_charges = df.total_charges.astype(float)
    
    return df

# Generic splitting function for continuous target.

def split_continuous(df):
    '''
    Takes in a df
    Returns train, validate, and test DataFrames
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, 
                                        test_size=.2, 
                                        random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, 
                                   test_size=.3, 
                                   random_state=123)

    # Take a look at your split datasets

    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
    return train, validate, test