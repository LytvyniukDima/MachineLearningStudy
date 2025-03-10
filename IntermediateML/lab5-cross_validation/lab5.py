# Set up code checking
from learntools.core import binder

binder.bind(globals())
from learntools.ml_intermediate.ex5 import *

print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Read the data
train_data = pd.read_csv("../input/train.csv", index_col="Id")
test_data = pd.read_csv("../input/test.csv", index_col="Id")

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = train_data.SalePrice
train_data.drop(["SalePrice"], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [
    cname
    for cname in train_data.columns
    if train_data[cname].dtype in ["int64", "float64"]
]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

my_pipeline = Pipeline(
    steps=[
        ("preprocessor", SimpleImputer()),
        ("model", RandomForestRegressor(n_estimators=50, random_state=0)),
    ]
)

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(
    my_pipeline, X, y, cv=5, scoring="neg_mean_absolute_error"
)

print("Average MAE score:", scores.mean())


def get_score(n_estimators):
    score_pipeline = Pipeline(
        steps=[
            ("preprocessor", SimpleImputer()),
            ("model", RandomForestRegressor(n_estimators=n_estimators, random_state=0)),
        ]
    )

    scores = -1 * cross_val_score(
        score_pipeline, X, y, cv=3, scoring="neg_mean_absolute_error"
    )

    return scores.mean()


# Check your answer
step_1.check()

results = {n_estimators: get_score(n_estimators) for n_estimators in range(50, 401, 50)}

step_2.check()


plt.plot(list(results.keys()), list(results.values()))
plt.show()

# Find the key with the maximum value in the results dictionary
n_estimators_best = min(results, key=results.get)

step_3.check()

print(f"The key with the minimum value is: {n_estimators_best}")
print(f"The minimum value is: {results[n_estimators_best]}")
