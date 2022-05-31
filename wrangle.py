import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import acquire_zillow

###################### Prep Zillow Data ######################

def wrangle_zillow(df):
    
    # show only data from the selected columns
    df1 = df[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips']]
     
    # Display readable summary statistics for numeric columns. Why isn't exam3 showing up?
    df1.describe().T

    # Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with any Null values, assign to df, and verify.
    df1 = df1.dropna()

    # Change all column data tyes to int64, reassign to df, and verify.
    df1 = df1.astype('int')

    df1.describe().T

    plt.figure(figsize=(21, 2))

# List of columns
cols = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt','yearbuilt','taxamount','fips']

for i, col in enumerate(cols):

    # i starts at 0, but plot nos should start at 1
    plot_number = i + 1 

    # Create subplot.
    plt.subplot(1,7, plot_number)

    # Title with column name.
    plt.title(col)

    # Display histogram for column.
    df1[col].hist(bins=10)

    # Hide gridlines.
    plt.grid(False)

    return df