import pandas as pd
import numpy as np
import os
import acquire 

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# *****************************************************************************************************************
# Not my function I found this and thought it would help with my analysis
#******************************************************************************************************************

# add total of duplicated values
def duplicated_or_missing(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows, 
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    missing_val = df.isnull().sum()
        # Percentage of missing values
    missing_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    missing_table = pd.concat([missing_val, missing_percent], axis=1)
        # Rename the columns
    missing_table_ren_columns = missing_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    missing_table_ren_columns = missing_table_ren_columns[
    missing_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(missing_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return missing_table_ren_columns

# ******************************************** ZILLOW ********************************************

def clean_zillow (df):

    '''
    Takes in a df and drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800, and bedroomcnt, yearbuilt, fips are changed to
    int.
    Return a clean df
    '''
    
    # drop duplicates
    df = df.drop_duplicates()
    #drop nulls
    df = df.dropna(how='any',axis=0)

    #drop all houses with bath = 0 and bedromms = 0
    #get the index to drop the rows
    ind = list(df[(df.bedroomcnt == 0) & (df.bathroomcnt == 0)].index)
    #drop
    df.drop(ind, axis=0, inplace= True)


    #drop all houses calculatedfinisheedsqf <800
    #get the index to drop
    lis =list(df[df['calculatedfinishedsquarefeet'] < 800].index)
    #drop the rows
    df.drop(lis, axis=0, inplace = True)

    #bedrooms, yearbuilt and fips can be converted to int
    df[['bedroomcnt', 'yearbuilt', 'fips']] = df[['bedroomcnt', 'yearbuilt', 'fips']].astype(int)
    return df



def wrangle_zillow():
    ''''
    This function will acquire zillow db using get_new_zillow function. then it will use another
    function named  clean_zillwo that drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800.
     bedroomcnt, yearbuilt, fips are changed to int.
    return cleaned zillow DataFrame
    '''
    df = acquire.get_new_zillow()
    zillow_df = clean_zillow(df)
    return zillow_df




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






def wrangle_student_math(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(
        df, "G3"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )