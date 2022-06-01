# Acquire Zillow Data

# OS needed to do local inspection of cache, to see if data exists locally
import os
# env.py contains our credentials to access the SQL server we are pulling the data from
from env import host, user, password
# Pandas is needed to perform SQL interaction
import pandas as pd
import numpy as np

def get_db_url(db_name, username=user, hostname=host, password=password):
    '''
    This function requires a database name (db_name) and uses the imported username,
    hostname, and password from an env file. 
    A url string is returned using the format required to connect to a SQL server.
    '''
    url = f'mysql+pymysql://{username}:{password}@{host}/{db_name}'
    return url


def acquire_zillow(use_cache = True):
    '''
    This function returns a dataframe that is constructed from a SQL query gathering Zillow data. It checks if there is already a local csv file
    with the Zillow data, and will create one if none is present. (This gathers select columns)
    '''
    # Checking to see if data already exists in local csv file
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    # If data is not local we will acquire it from SQL server
    print('Acquiring data from SQL db')
    # Query to refine what data we want to grab
    query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet, fips, yearbuilt, taxvaluedollarcnt
    FROM properties_2017
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    JOIN predictions_2017 USING(parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    AND transactiondate LIKE "2017%%"
    '''
    # Acquisition and creation of dataframe
    df = pd.read_sql(query, get_db_url('zillow'))
    # Renaming columns 
    df = df.rename(columns =   {'bedroomcnt': 'bedrooms',
                                'bathroomcnt': 'bathrooms',
                                'calculatedfinishedsquarefeet': 'house_area',
                                'lotsizesquarefeet': 'lot_area',
                                'taxvaluedollarcnt': 'tax_value'})    
    # Creation of csv file
    df.to_csv('zillow.csv', index=False)
    # Returning the df
    return df