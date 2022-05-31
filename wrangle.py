import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import acquire_zillow

###################### Prep Zillow Data ######################

def wrangle_zillow(cached=True):
    '''
    This function reads in iris data from Codeup database if cached == False
    or if cached == True reads in iris df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow.csv') == False:
        df = acquire_zillow()
    else:
        df = pd.read_csv('zillow.csv', index_col=0)
   
    # show only data from the selected columns
        df= df[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips']]
     
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
