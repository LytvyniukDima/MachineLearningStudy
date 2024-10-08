import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

def adjust_dataset(input_data):
    input_data.pop('Embarked')
    input_data.pop('Cabin')
    input_data.pop('Name')
    input_data.pop('Ticket')
    
    input_data['Sex'] = input_data['Sex'].map(
        {
            'male': 0,
            'female': 1
        }
    )

    input_data = fill_na_with_mean(input_data, 'Age')

    return input_data


def create_sequential_model(input_data):
    X = test_data.copy()
    y = X.pop('Survived')

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

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
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
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
    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()
    print(("Best Validation Loss: {:0.4f}" +\
        "\nBest Validation Accuracy: {:0.4f}")\
        .format(history_df['val_loss'].min(), 
                history_df['val_binary_accuracy'].max()))

    plt.show()

    return model


def prepare_rainforest_data(test_data):
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(test_data[features])

    return X

def create_rainforest_model(test_data):
    y = test_data['Survived']

    X = prepare_rainforest_data(test_data)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)

    return model

test_data = pd.read_csv('./input/train.csv')

test_data = adjust_dataset(test_data)
test_data.pop('PassengerId')

# model = create_sequential_model(test_data)
model = create_rainforest_model(test_data)

predictions_data = pd.read_csv('./input/test.csv')

predictions_data = adjust_dataset(predictions_data)
passenger_ids = predictions_data.pop('PassengerId').to_numpy()
X_predictions = prepare_rainforest_data(predictions_data)

predictions = model.predict(X_predictions)
integer_predictions = np.round(predictions).flatten()
print(integer_predictions)

output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': integer_predictions})
output.to_csv('submission.csv', index=False)
