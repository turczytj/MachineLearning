# http://www.pycaret.org/tutorials/html/CLF101.html

# NOTE: The code below is not running correctly based on the web link above. I believe it must run in JupyterLab and
#       not from within an IDE. See link for details.

from pycaret.datasets import get_data
from pycaret.classification import *


def run_demo():
    # Note: this required adding the line "from IPython.display import display" to the file
    #   C:\ProgramData\Anaconda3\Lib\site-packages\pycaret\datasets.py
    dataset = get_data('credit')

    # Check the shape of data
    dataset.shape

    # In order to demonstrate the predict_model() function on unseen data, a sample of 1200 records has been withheld
    # from the original dataset to be used for predictions. This should not be confused with a train/test split as this
    # particular split is performed to simulate a real life scenario. Another way to think about this is that these
    # 1200 records are not available at the time when the machine learning experiment was performed.
    data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
    data_unseen = dataset.drop(data.index).reset_index(drop=True)

    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))

    # The setup() function initializes the environment in pycaret and creates the transformation pipeline to prepare the
    # data for modeling and deployment. setup() must be called before executing any other function in pycaret. It takes
    # two mandatory parameters: a pandas dataframe and the name of the target column.
    exp_clf101 = setup(data=data, target='default', session_id=123)

    # Comparing all models to evaluate performance is the recommended starting point for modeling once the setup is
    # completed (unless you exactly know what kind of model you need, which is often not the case). This function trains
    # all models in the model library and scores them using stratified cross validation for metric evaluation. The
    # output prints a score grid that shows average Accuracy, AUC, Recall, Precision, F1 and Kappa accross the folds
    # (10 by default) of all the available models in the model library.
    compare_models()