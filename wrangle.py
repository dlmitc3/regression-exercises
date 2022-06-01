# Wrangling Zillow Data

# os needed to do local inspection of cache, to see if data exists locally
import os
# env.py contains our credentials to access the SQL server we are pulling the data from
from env import host, user, password
# Pandas is needed to perform SQL interaction
import pandas as pd
import numpy as np
# Splitting function
from sklearn.model_selection import train_test_split
# Imputer
from sklearn.impute import SimpleImputer


# Acquire Functions

def get_db_url(db_name, username=user, hostname=host, password=password):
    '''
    This function requires a database name (db_name) and uses the imported username,
    hostname, and password from an env file. 
    A url string is returned using the format required to connect to a SQL server.
    '''
    url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    return url


def acquire_zillow_data(use_cache= True):
    '''
    Acquire the zillow data using SQL query and get_db_url() with credentials from env.py
    '''
    # Checking to see if data already exists in local csv file
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    # If data is not local we will acquire it from SQL server
    print('Acquiring data from SQL db')
    # Query to refine what data we want to grab 
    # bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips from properties_2017 
    # where propertylandusetypeid == 261 (Single Family Residential)
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261 OR propertylandusetypeid = 279 
    '''
    # Command line interaction with SQL server and assignment to dataframe (df)
    df = pd.read_sql(query, get_db_url('zillow'))
    # Ranaming columns
    df = df.rename(columns = {'bedroomcnt': 'bedrooms',
                              'bathroomcnt': 'bathrooms',
                              'calculatedfinishedsquarefeet': 'area',
                              'taxvaluedollarcnt': 'tax_value',
                              'yearbuilt': 'year_built'})
    # Creation of csv file
    df.to_csv('zillow.csv', index=False)
    # Returns the dataframe
    return df


# Preparation and Splitting

def remove_outliers(df, k, col_list):
    '''
    Removes outliers from a list of columns in df, and returns the df.
    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75]) # get quartiles
        iqr = q3 - q1 # calculate interquartile range
        upper_bound = q3 + k * iqr # upper bound
        lower_bound = q1 - k * iqr # lower bound
        # Remove your outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    # Return df
    return df


def prep_zillow(df):
    '''
    Takes in a dataframe and prepares the data. 
    Returns train, validate, and test dataframes after splitting the data.
    '''
    # Remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])

    # Ensuring that the '0' remains for the fips column after converting to str type
        # Convert to an int first to remove the trailing decimal
    df.fips = df.fips.astype(int)
    df.fips = df.fips.astype(str)
    df.fips = '0' + df.fips
    
    # Split our df in to train, validate, and test splits (3 df)
    train, validate, test = zillow_split(df)

    # Impute year built using mode
    imputer = SimpleImputer(strategy='median')
    # Fit on train df
    imputer.fit(train[['year_built']])
    # Apply imputer to your splits
    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])

    # Return train, validate, test splits
    return train, validate, test


def zillow_split(df):
    '''
    This function takes in a dataframe and returns train, validate, test splits. (dataframes)
    An initial 20% of data is split to place as 'test'
    A second split is performed (on the remaining 80%) between train and validate (70/30)
    '''
    # First split with 20% going to test
    train_validate, test = train_test_split(df, train_size = .8, 
                                                 random_state = 123)
    # Second split with 70% of remainder going to train, 30% to validate
    train, validate = train_test_split(train_validate, train_size = .7,
                                                random_state=123)
    # Return train, validate, test (56%, 24%, 20% splits of original df)
    return train, validate, test


def wrangle_zillow():
    '''
    Using acquire and preparation functions to 'wrangle' the zillow data. Returns train, validate, and test dataframes.
    '''
    # Acquire the data
    df = acquire_zillow_data()
    # Prep the data (returns train, validate, test)
    train, validate, test = prep_zillow(df)
    # Return split dataframes
    return train, validate, test