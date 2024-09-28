# %%
import numpy as np
import pandas as pd

def impute_nan_value(df, variable):
    df[variable+"_random"]=df[variable]
    ##It will have the random sample to fill the na
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    ##pandas need to have same index in order to merge the dataset
    random_sample.index=df[df[variable].isnull()].index #replace random_sample index with NaN values index
    #replace where NaN are there
    df.loc[df[variable].isnull(),variable+'_random']=random_sample
    df[variable]=df[variable+"_random"]
    df = df.drop(variable+"_random",axis=1)
    return df.copy()


