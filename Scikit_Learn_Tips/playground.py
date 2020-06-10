# https://medium.com/@simonprdhm/9-scikit-learn-tips-for-data-scientist-2a84ffb385ba

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def run_demo():
    ####################################################################################################################
    # Tip 1: Use make_column_transformer to apply different preprocessing to different columns                         #
    # NOTE: I'm not sure this works                                                                                    #
    ####################################################################################################################

    # Load data (loading Titanic dataset)
    data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

    # Make Transformer
    preprocessing = make_column_transformer(
                        (OneHotEncoder(), ['Pclass', 'Sex']),
                        (SimpleImputer(), ['Age']),
                        remainder='passthrough')

    # Fit-Transform data with transformer
    data = preprocessing.fit_transform(data)

    ####################################################################################################################
    # Tip 2: Use make_column_selector to change data types to different columns                                        #
    # NOTE: I'm not sure this works                                                                                    #
    ####################################################################################################################

    # Load data (loading Titanic dataset)
    data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

    # Make Transformer
    preprocessing = make_column_transformer(
        (OneHotEncoder(), make_column_selector(dtype_include='object')),
        (SimpleImputer(), make_column_selector(dtype_include='int')),
        remainder='drop'
    )

    # Fit-Transform data with transformer
    data = preprocessing.fit_transform(data)

    ####################################################################################################################
    # Tip 3: Use Pipeline. Pipeline chains together multiple preprocessing steps. The output of each step is used as   #
    # input to the next step, it makes it easy to apply the same preprocessing to Train and Test.                      #
    ####################################################################################################################

    # Load data (loading Titanic dataset)
    data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

    # Set X and y
    X = data.drop('Survived', axis=1)
    y = data[['Survived']]

    # Split Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Set variables
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
    imputer = SimpleImputer(add_indicator=True, verbose=1)
    scaler = StandardScaler()
    clf = DecisionTreeClassifier()

    # Make Transformer
    preprocessing = make_column_transformer(
        (make_pipeline(imputer, scaler), ['Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare'])
        , (ohe, ['Pclass', 'Sex', 'Name'])
        , remainder='passthrough')

    # Make pipeline
    pipe = make_pipeline(preprocessing, clf)

    # Fit model
    pipe.fit(X_train, y_train.values.ravel())
    print("Best score : %f" % pipe.score(X_test, y_test.values.ravel()))

    ####################################################################################################################
    # Tip 4: You can grid search an entire pipeline and fine optimal tuning parameters                                 #
    ####################################################################################################################

    # Load data (loading Titanic dataset)
    data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

    # Set X and y
    X = data.drop('Survived', axis=1)
    y = data[['Survived']]

    # Set variables
    clf = LogisticRegression()
    ohe = OneHotEncoder()
    scaler = StandardScaler()
    imputer = SimpleImputer()

    # Make Transformer
    preprocessing = make_column_transformer(
        (make_pipeline(imputer, scaler), ['Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']),
        (ohe, ['Sex']), remainder='drop')

    # Make pipeline
    pipe = make_pipeline(preprocessing, clf)

    # Set params for Grid Search
    params = {}
    params['logisticregression__C'] = [0.1, 0.2, 0.3]
    params['logisticregression__max_iter'] = [200, 500]

    # Run grid search
    grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
    grid.fit(X, y.values.ravel())

    print(grid.best_score_)
    print(grid.best)

