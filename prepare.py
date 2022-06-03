# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing




# This function takes in the df generated via acquire.zillow_data function and prepares the data to be split into train, validate & test dataframes 

def prep_zillow(df):
    ##### Format transactiondate
    # Replace '-' with '' and Convert transactiondate to float
    df['transactiondate'] = df['transactiondate'].str.replace("-", "")
    # Convert to float
    df['transactiondate'] = df['transactiondate'].astype(float)
    # Convert float back to datetime format
    df['transactiondate'] = pd.to_datetime(df['transactiondate'], format='%Y%m%d')
    # drop duplicated columns using unique key 'parcelid'.  No new information in duplicated entries
    df = df.drop_duplicates(subset=['parcelid'])
    df = df.dropna()
    # rename columns
    df = df.rename(columns={'bathroomcnt': 'baths', 'calculatedfinishedsquarefeet': 'sqft', 'bedroomcnt': 'beds', 'taxvaluedollarcnt':'value', 'taxamount':'taxes', 'propertylandusetypeid':'useid'})
    # drop parcelid
    df = df.drop(columns=('parcelid'))
    return df   

def fips_conversion(df):
    #convert fips to integer to drop decimals before converting to string
    df['fips'] = df['fips'].astype(int)
    # convert fips data to string so that we can split
    df['fips'] = df['fips'].astype('str')
    # Create State column by stripping state reference location(first two digits)
    df['state'] = df['fips'].str[:1]
    # Create County column by stripping state reference location(last three digits)
    df['county'] = df['fips'].str[1:]
    # Replace numerics with labels
    df['state'] = df['state'].replace("6", "California")
    df['county'] = df['county'].replace({'037':'Los Angeles', '059':'Orange', '111':'Ventura'})
    return df

def calculate_tax_rate(df):
    df['tax_rate'] = round((df.taxes / df.value), 3)
    # drop outliers
    df = df.drop(df[df['tax_rate']>0.3].index)
    return df

# Train, Validate, Test data split


def split_zillow(df, target):
    '''
    this function takes in the zillow dataframe
    splits into train, validate and test subsets
    then splits for X (features) and y (target)
    '''
    # split df into 20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    # split train_validate into 30% validate, 70% train
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    # Split with X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def robust_scale():
    '''this function will:
        * set the scaler object
        * fit_transform X_train numerics
        * stacknon-numeric columns to array
        * transform numpy array to pandas dataframe
        * assign labels back to dataframe
        # return X_train_scaled_df, X_validate_scaled_df, X_test_scaled_df
    '''
    # Set the scaler object
    scaler = sklearn.preprocessing.RobustScaler()
    # Fit to a merged array
    X_train_scaled = np.column_stack((scaler.fit_transform(X_train[['baths', 'beds', 'sqft', 'taxes', 'tax_rate']]),X_train[['fips', 'yearbuilt', 'useid', 'state', 'county']]))
    X_validate_scaled = np.column_stack((scaler.fit_transform(X_validate[['baths', 'beds', 'sqft', 'taxes', 'tax_rate']]),X_validate[['fips', 'yearbuilt', 'useid', 'state', 'county']]))
    X_test_scaled = np.column_stack((scaler.fit_transform(X_test[['baths', 'beds', 'sqft', 'taxes', 'tax_rate']]),X_test[['fips', 'yearbuilt', 'useid', 'state', 'county']]))\
    #convert numpy arrays into dataframes
    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_validate_scaled_df = pd.DataFrame(X_validate_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)
    # assign names back to columns in dataframe
    X_train_scaled_df.columns=['baths', 'beds', 'sqft', 'taxes', 'tax_rate', 'fips', 'yearbuilt', 'useid', 'state', 'county']
    X_validate_scaled_df.columns=['baths', 'beds', 'sqft', 'taxes', 'tax_rate', 'fips', 'yearbuilt', 'useid', 'state', 'county']
    X_test_scaled_df.columns=['baths', 'beds', 'sqft', 'taxes', 'tax_rate', 'fips', 'yearbuilt', 'useid', 'state', 'county']
    
    return X_train_scaled_df, X_validate_scaled_df, X_test_scaled_df


def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedroomcnt', 'bathroomcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
