import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import env
import wrangle


def plot_variable_pairs(df):
    '''this function will take any any df and pairplot all the features as well as plot a regression line in red'''
    sns.pairplot(df[df.columns.tolist()], corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.show()

def months_to_years(df):
    '''function takes in df from wrangle_telco(), adds a tenure_years column that divides tenure/12 and formats it to a whole number'''
    df['tenure_years'] = (df['tenure']/12).map('{:,.0f}'.format)
    return df