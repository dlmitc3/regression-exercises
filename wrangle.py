import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import acquire_zillow

###################### Prep Zillow Data ######################

def wrangle_zillow(df):
     '''
    SQL Zillow data base information into a pandas DataFrame,
    created df1 file with only columns requested, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned data to DataFrame.
    '''
    # Acquire data from csv file.
    df = acquire.acquire_zillow()

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

    df.describe().T

    return df
