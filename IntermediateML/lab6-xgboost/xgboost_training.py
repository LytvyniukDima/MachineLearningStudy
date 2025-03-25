import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

# Select subset of predictors
cols_to_use = ["Rooms", "Distance", "Landsize", "BuildingArea", "YearBuilt"]
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

print(X_train.head())

my_model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_rate=0.05)
my_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,
)

print(my_model)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
