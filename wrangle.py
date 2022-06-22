import pandas as pd
import numpy as np
import os
import acquire 

from sklearn.model_selection import train_test_split
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# ****************************  connection **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, username, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'

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

# ***************************************************************************************************
#                                     ZILLOW DB
# ***************************************************************************************************

#acquire data for the first time
def get_new_zillow():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with columns :
     bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
    '''
    sql_query = '''
    SELECT parcelid, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,
    taxvaluedollarcnt
    FROM properties_2017
    JOIN predictions_2017 as pred USING (parcelid)
    WHERE pred.transactiondate >= '2017-05-01' AND pred.transactiondate <= '2017-08-31'
    AND propertylandusetypeid > 259 AND propertylandusetypeid  < 266;
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))

#acquire data main function 
def get_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df





def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test

def split_Xy (train, validate, test, df.taxvaluedollarcnt):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [df.taxvaluedollarcnt])
    y_train= train[df.taxvaluedollarcnt]
    #split validate
    X_validate = validate.drop(columns= [df.taxvaluedollarcnt])
    y_validate= validate[df.taxvaluedollarcnt]
    #split validate
    X_test = test.drop(columns= [df.taxvaluedollarcnt])
    y_test= test[df.taxvaluedollarcnt]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test



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




def miss_dup_values(df):
    '''
    this function takes a dataframe as input and will output metrics for missing values and duplicated rows, 
    and the percent of that column that has missing values and duplicated rows
    '''
        # Total missing values
    mis_val = df.isnull().sum()
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        #total of duplicated
    dup = df.duplicated().sum()  
        # Percentage of missing values
    dup_percent = 100 * dup / len(df)
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
           "There are " + str(mis_val_table_ren_columns.shape[0]) +
           " columns that have missing values.")
    print( "  ")
    print (f"** There are {dup} duplicate rows that represents {round(dup_percent, 2)}% of total Values**")
        # Return the dataframe with missing information
    return mis_val_table_ren_columns


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