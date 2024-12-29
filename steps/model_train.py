import logging

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client
from sklearn.base import RegressorMixin

from config import ModelNameConfig
from src.model_dev import LinearRegressionModel

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train : pd.DataFrame,
    X_test : pd.DataFrame,
    y_train : pd.DataFrame,
    y_test : pd.DataFrame,
    config : str = ModelNameConfig.model_name
 ) ->RegressorMixin:
    '''
    trains the model on the ingested data

    Args:
        
    '''
    try:
        model = None
        if ModelNameConfig.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(ModelNameConfig.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e