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

# Convolutional Neural Networks, or CNNs for short, are a type of network designed for image input.
# They are comprised of models with convolutional layers that extract features (called feature maps)
# and pooling layers that distill features down to the most salient elements.
#
# CNNs are most well-suited to image classification tasks, although they can be used on a wide array
# of tasks that take images as input.
#
# A popular image classification task is the MNIST handwritten digit classification. It involves tens
# of thousands of handwritten digits that must be classified as a number between 0 and 9.
#
# The tf.keras API provides a convenience function to download and load this dataset directly.
def create_cnn_for_image_classification():
    from numpy import unique
    from numpy import argmax
    from tensorflow.keras.datasets.mnist import load_data
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dropout

    # load dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Note that the images are arrays of grayscale pixel data; therefore, we must add a channel dimension
    # to the data before we can use the images as input to the model. The reason is that CNN models expect
    # images in a channels-last format, that is each example to the network has the dimensions
    # [rows, columns, channels], where channels represent the color channels of the image data.

    # reshape data to have a single channel
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    # determine the shape of the input images
    in_shape = x_train.shape[1:]

    # determine the number of classes
    n_classes = len(unique(y_train))
    print(in_shape, n_classes)

    # It is a good idea to scale the pixel values from the default range of 0-255 to 0-1 when training a CNN
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # define model
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=in_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    # define loss and optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # fit the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=0)

    # evaluate the model
    loss, model_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Accuracy: %.3f' % model_accuracy)

    # make a prediction
    image = x_train[0]
    yhat = model.predict([[image]])
    print('Predicted: class=%d' % argmax(yhat))

    return model_accuracy
