import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

def fill_na_with_mean(df, column_name):
    column_mean = df[column_name].mean()

  # Fill missing values with the mean
    df[column_name] = df[column_name].fillna(column_mean)
    
    return df

def remove_na_values(df, column_name):
    df = test_data.dropna(subset=[column_name])
    return df


test_data = pd.read_csv('./input/train.csv')

print(test_data.head())

print(test_data.columns)

test_data.pop('Embarked')
test_data.pop('Cabin')
test_data.pop('Name')
test_data.pop('Ticket')
test_data.pop('PassengerId')

print('-------')
print(test_data.head())

test_data['Sex'] = test_data['Sex'].map(
    {
        'male': 0,
        'female': 1
    }
)

print('-------')
print(test_data.head())

test_data = fill_na_with_mean(test_data, 'Age')

X = test_data.copy()
y = X.pop('Survived')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

X_valid, X_predict, y_valid, y_predict = train_test_split(X_valid, y_valid, stratify=y_valid, train_size=0.75)

input_shape = [X_train.shape[1]]

model = keras.Sequential([
    layers.Dense(240, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(120, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(60, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(30, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='mae',
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_plot = history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));

plt.show()

predictions = model.predict(X_predict)
integer_predictions = np.round(predictions)

# Do something with the predictions
print(integer_predictions)
print(X_valid.head())