# Code was found at:
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

def get_tf_version():
    import tensorflow as tf

    return tf.__version__

# A Multilayer Perceptron (MLP) model is comprised of layers of nodes where each node is connected to all outputs
# from the previous layer and the output of each node is connected to all inputs for nodes in the next layer.
#
# An MLP is created with one or more Dense layers. This model is appropriate for tabular data, that is data
# as it looks in a table or spreadsheet with one column for each variable and one row for each row. There are
# three predictive modeling problems you may want to explore with an MLP; they are binary classification,
# multiclass classification, and regression.

# Use the Ionosphere binary (two-class) classification dataset to demonstrate an Multilayer Perceptron (MLP) for
# binary classification.
def create_mlp_for_binary_classification():
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    # load the dataset
    ionosphere_dataset_path = '.\KerasToTensorFlow\ionosphere.csv'
    ionosphere_df = read_csv(ionosphere_dataset_path, header=None)

    # split dataset into input and output columns
    X, y = ionosphere_df.values[:, :-1], ionosphere_df.values[:, -1]

    # ensure all data are floating point values
    X = X.astype('float32')

    # encode strings to integer
    y = LabelEncoder().fit_transform(y)

    # split dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # determine the number of input features
    num_input_features = X_train.shape[1]

    # define model
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(num_input_features,)))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # evaluate the model
    loss, model_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % model_accuracy)

    # make a prediction
    row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
    yhat = model.predict([row])
    print('Predicted: %.3f' % yhat)

    return model_accuracy

# Use the Iris  classification dataset to demonstrate an Multilayer Perceptron (MLP) for multiclass classification.
#
# Given that it is a multiclass classification, the model must have one node for each class in the output layer
# and use the softmax activation function. The loss function is the ‘sparse_categorical_crossentropy‘, which is
# appropriate for integer encoded class labels (e.g. 0 for one class, 1 for the next class, etc.).
def create_mlp_for_multiclass_classification():
    from numpy import argmax
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    # load the dataset
    iris_dataset_path = '.\KerasToTensorFlow\iris.csv'
    iris_df = read_csv(iris_dataset_path, header=None)

    # split into input and output columns
    X, y = iris_df.values[:, :-1], iris_df.values[:, -1]

    # ensure all data are floating point values
    X = X.astype('float32')

    # encode strings to integer
    y = LabelEncoder().fit_transform(y)

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    num_input_features = X_train.shape[1]

    # define model
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(num_input_features,)))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(3, activation='softmax'))

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # evaluate the model
    loss, model_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % model_accuracy)

    # make a prediction
    row = [5.1,3.5,1.4,0.2]
    yhat = model.predict([row])
    print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

    return model_accuracy

# This is a regression problem that involves predicting a single numerical value. As such, the output layer
# has a single node and uses the default or linear activation function (no activation function).
# The mean squared error (MSE) loss is minimized when fitting the model.
#
# Recall that this is a regression, not classification; therefore, we cannot calculate classification accuracy.
#
# Use the Boston housing dataset to demonstrate an Multilayer Perceptron (MLP) for regression predictive modeling.
def create_mlp_for_regression_predictions():
    from numpy import sqrt
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    boston_dataset_path = '.\\KerasToTensorFlow\\boston.csv'
    boston_df = read_csv(boston_dataset_path, header=None)

    # split into input and output columns
    X, y = boston_df.values[:, :-1], boston_df.values[:, -1]

    # encode strings to integer
    y = LabelEncoder().fit_transform(y)

    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # determine the number of input features
    num_input_features = X_train.shape[1]

    # define model
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(num_input_features,)))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))

    # compile the model
    model.compile(optimizer='adam', loss='mse')

    # fit the model
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # evaluate the model
    error = model.evaluate(X_test, y_test, verbose=0)
    print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

    return model

def make_mlp_regression_prediction(model, data):
    yhat = model.predict([data])
    print('Predicted: %.3f' % yhat)

    return yhat
