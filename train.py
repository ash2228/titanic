import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from clean_data import clean_ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    ds = pd.read_csv("Titanic-Dataset.csv")

    ds = clean_ds(ds=ds)
    model = RandomForestClassifier(n_estimators=5000)
    features = ds.drop(["Survived", "Ticket", "Name", "PassengerId", "Embarked"], axis=1)
    target = ds["Survived"]
    xTrain, xTest, yTrain, yTest = train_test_split(features, target, train_size=0.8)
    model.fit(xTrain, yTrain)
    pred = model.predict(xTest)
    accuracy = accuracy_score(yTest, pred)
    print(accuracy)