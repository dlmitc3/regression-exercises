import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import acquire_zillow

###################### Prep Zillow Data ######################

def wrangle_zillow(cached=True):
    '''
    This function reads in Zillow data from Codeup database if cached == False
    or if cached == True reads in Zillow df from a csv file, returns df
    '''

     # use my aquire function to read data into a df from a csv file
    df = acquire_zillow(cached)
    
    # show only data from the selected columns
    df = df[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips']]
     
    # Display readable summary statistics for numeric columns.
    df.describe().T

    # Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df1.
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with any Null values, assign to df1, and verify.
    df = df.dropna()

    # Change all column data tyes to int64, reassign to df1, and verify.
    df = df.astype('int')

    # describe data
    df.describe().T

    return df

