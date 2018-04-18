import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score


def get_cross_validation_score():

    data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv",index_col = 0)
    #print(data.head())


    feature_cols = ['TV','radio','newspaper']
    X = data[feature_cols]
    y = data.sales
    lm_sales = LinearRegression()


    scores = cross_val_score(lm_sales,X,y,cv = 10, scoring = 'mean_squared_error')
    #print scores
    return str(-scores.mean())

#print (get_cross_validation_score())
