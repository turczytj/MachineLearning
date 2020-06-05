# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
from numpy import mean


def create_imbalanced_binary_classification():
    """
    Generate and plot a synthetic imbalanced classification dataset
    """

    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99],
                               flip_y=0, random_state=1)

    # Summarize class distribution
    counter = Counter(y)
    print(counter)

    # Scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    pyplot.legend()
    pyplot.show()


def run_smote_oversampling():
    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

    # Summarize class distribution
    counter = Counter(y)
    print(counter)

    # Transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    # Summarize the new class distribution
    counter = Counter(y)
    print(counter)

    # Scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    pyplot.legend()
    pyplot.show()


def run_smote_oversampling_and_undersampling():
    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

    # Summarize class distribution
    counter = Counter(y)
    print(counter)

    # Define pipeline
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    # Transform the dataset
    X, y = pipeline.fit_resample(X, y)

    # Summarize the new class distribution
    counter = Counter(y)
    print(counter)

    # Scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))

    pyplot.legend()
    pyplot.show()


def run_smote_to_evaluate_imbalanced_decision_tree():
    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    # Define pipeline
    steps = [('over', SMOTE()), ('model', DecisionTreeClassifier())]
    pipeline = Pipeline(steps=steps)

    # Evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % mean(scores))


def run_smote_to_evaluate_balanced_decision_tree():
    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

    # Define pipeline
    steps = [('over', SMOTE()), ('model', DecisionTreeClassifier())]
    pipeline = Pipeline(steps=steps)

    # Evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % mean(scores))


def run_smote_evaluate_oversampling_and_undersampling_decision_tree():
    # Define dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

    # Define pipeline
    model = DecisionTreeClassifier()
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('over', over), ('under', under), ('model', model)]
    pipeline = Pipeline(steps=steps)

    # Evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % mean(scores))


def run_main():
    create_imbalanced_binary_classification()
    run_smote_oversampling()
    run_smote_oversampling_and_undersampling()
    run_smote_to_evaluate_imbalanced_decision_tree()
    run_smote_to_evaluate_balanced_decision_tree()
    run_smote_evaluate_oversampling_and_undersampling_decision_tree()

    # NOTE: there were additional ways documented in link to use SMOTE
