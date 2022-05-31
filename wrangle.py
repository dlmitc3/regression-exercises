import pandas as pd
import os
from env import host, user, password



def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

###################### Acquire Zillow Data ######################

def new_wrangle_data():
    '''
    This function reads the Zillow data from the Zillow.csv file..
    '''
    sql_query = """
                SELECT * from properties_2017
                join predictions_2017 using (id)
                join properties_2016 using (id)
                join predictions_2016 using (id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_df.csv')
    return df


