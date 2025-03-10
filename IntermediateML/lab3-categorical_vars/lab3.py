# Set up code checking
from learntools.core import binder

binder.bind(globals())
from learntools.ml_intermediate.ex3 import *

print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


def create_submission(model, final_X_test):
    preds = model.predict(final_X_test)

    # Save test predictions to file
    output = pd.DataFrame({"Id": final_X_test.index, "SalePrice": preds})
    output.to_csv("submission.csv", index=False)


def create_OH_submission():
    X_test = pd.read_csv("../input/test.csv", index_col="Id")
    X = pd.read_csv("../input/train.csv", index_col="Id")

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=["SalePrice"], inplace=True)
    y_train = X.SalePrice
    X.drop(["SalePrice"], axis=1, inplace=True)

    # To keep things simple, we'll drop columns with missing values
    cols_with_missing = [
        col for col in X.columns if X[col].isnull().any() or X_test[col].isnull().any()
    ]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # All categorical columns
    object_cols = [col for col in X.columns if X[col].dtype == "object"]

    # Columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if set(X[col]) == set(X_test[col])]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    X.drop(bad_label_cols, axis=1)
    X_test.drop(bad_label_cols, axis=1)

    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    low_cardinality_X_train = X.drop(high_cardinality_cols, axis=1)
    low_cardinality_X_Test = X_test.drop(high_cardinality_cols, axis=1)

    OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    OH_cols_train = pd.DataFrame(
        OH_encoder.fit_transform(low_cardinality_X_train[low_cardinality_cols])
    )
    OH_cols_test = pd.DataFrame(
        OH_encoder.transform(low_cardinality_X_Test[low_cardinality_cols])
    )

    # One-hot encoding removed index; put it back
    OH_cols_train.index = low_cardinality_X_train.index
    OH_cols_test.index = low_cardinality_X_Test.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = low_cardinality_X_train.drop(low_cardinality_cols, axis=1)
    num_X_test = low_cardinality_X_Test.drop(low_cardinality_cols, axis=1)

    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

    # Ensure all columns have string type
    OH_X_train.columns = OH_X_train.columns.astype(str)
    OH_X_test.columns = OH_X_test.columns.astype(str)

    final_model = RandomForestRegressor(n_estimators=100, random_state=0)
    final_model.fit(OH_X_train, y_train)

    create_submission(final_model, OH_X_test)


# Read the data
X = pd.read_csv("../input/train.csv", index_col="Id")
X_test = pd.read_csv("../input/test.csv", index_col="Id")

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X.SalePrice
X.drop(["SalePrice"], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

# Check your answers
step_1.check()

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

print(
    "Unique values in 'Condition2' column in training data:",
    X_train["Condition2"].unique(),
)
print(
    "\nUnique values in 'Condition2' column in validation data:",
    X_valid["Condition2"].unique(),
)

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print("Categorical columns that will be label encoded:", good_label_cols)
print("\nCategorical columns that will be dropped from the dataset:", bad_label_cols)

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(label_X_train[col])
    label_X_valid[col] = label_encoder.transform(label_X_valid[col])

# Check your answer
step_2.b.check()

print("MAE from Approach 2 (Label Encoding):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
print(sorted(d.items(), key=lambda x: x[1]))

# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25

# Check your answers
step_3.a.check()

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with a one-hot encoding?
OH_entries_added = 990000

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with a label encoding?
label_entries_added = 0

# Check your answers
step_3.b.check()

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

print("Categorical columns that will be one-hot encoded:", low_cardinality_cols)
print(
    "\nCategorical columns that will be dropped from the dataset:",
    high_cardinality_cols,
)

from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
low_cardinality_X_train = X_train.drop(high_cardinality_cols, axis=1)
low_cardinality_X_valid = X_valid.drop(high_cardinality_cols, axis=1)

OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
OH_cols_train = pd.DataFrame(
    OH_encoder.fit_transform(low_cardinality_X_train[low_cardinality_cols])
)
OH_cols_valid = pd.DataFrame(
    OH_encoder.transform(low_cardinality_X_valid[low_cardinality_cols])
)

# One-hot encoding removed index; put it back
OH_cols_train.index = low_cardinality_X_train.index
OH_cols_valid.index = low_cardinality_X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = low_cardinality_X_train.drop(low_cardinality_cols, axis=1)
num_X_valid = low_cardinality_X_valid.drop(low_cardinality_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

# Check your answer
step_4.check()

# print("MAE from Approach 3 (One-Hot Encoding):")
# print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# crete submission
create_OH_submission()
