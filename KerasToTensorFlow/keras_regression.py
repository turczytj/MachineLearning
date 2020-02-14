# Code came from https://towardsdatascience.com/getting-familiar-with-keras-dd17a110652d

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def calculate_mpg():
    auto_df = pd.read_csv('.\\KerasToTensorFlow\\auto-mpg.csv')

    X = np.array(auto_df[['Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']])
    y = np.array(auto_df['MPG'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

    # Reshape our array of labels
    y_train = np.reshape(y_train, (-1,1))

    # Now let’s define our model. Let’s start with a neural network with input and hidden layers with 64 neurons.
    # The input dimension is 7, the optimizer is ‘adam’ and the loss function is ‘mse’. We will also use 1000 epochs
    # and a batch size of 10
    model = Sequential()

    model.add(Dense(64, input_dim=7, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    model.fit(X_train, y_train, epochs=1000, batch_size=10)

    # Now, let’s visualize our results
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

