import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_ds(ds):
    cat_encoder = OneHotEncoder(sparse_output=False)
    newPclasses = cat_encoder.fit_transform(ds["Pclass"].to_numpy().reshape((-1, 1)))
    col_names = [f"Pclass_{int(i)}" for i in cat_encoder.categories_[0]]
    ds[col_names] = newPclasses
    ds.drop(["Pclass", "Cabin"], axis=1, inplace=True)
    ds["Sex"].replace({"male": 1, "female": 0}, inplace=True)
    return ds