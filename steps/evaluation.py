import logging
from typing import Tuple

import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client
from typing_extensions import Annotated 
from sklearn.base import RegressorMixin

from src.evaluation import MSE, RMSE, R2

experiment_tracker = Client().active_stack.experiment_tracker.name

@step(experiment_tracker=experiment_tracker)
def evaluate_model(model : RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   )->Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"]
                   ]:
    '''
    Evaluates the model trained on the ingested data

    Args:
        df : the ingested data
    '''
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calcualte_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        prediction = model.predict(X_test)
        rmse_class = RMSE()
        rmse = rmse_class.calcualte_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        prediction = model.predict(X_test)
        r2_class = R2()
        r2 = r2_class.calcualte_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)

        return r2, rmse
    except Exception as e:
        logging.error("eror in calculate scores : {}".format(e))
        raise e
    