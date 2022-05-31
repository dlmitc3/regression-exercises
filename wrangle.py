import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def wrangle_zillow(df):
    '''
    This function will drop all collumns except, bedroomcnt, bathroomcnt, 
    calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, 
    and fips from the zillow database for all 'Single Family Residential' properties. 
    '''
#drop duplicates columns
    df1 = df[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt', 'yearbuilt', 'taxamount', 'fips']]
  
    return df1