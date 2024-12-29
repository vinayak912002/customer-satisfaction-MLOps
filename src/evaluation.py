# to learn about the strategy pattern used in this file go to the data_cleaning.py file
import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    '''
    Abstract class defining the strategy for evaluation of models 
    '''
    @abstractmethod
    def calcualte_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        '''
        calculates the scores for the model
        
        Args:
            y_true: True  labels
            y_pred: predicteg labels

        returns:
            None
        '''
        pass

class MSE(Evaluation):
    '''
    Evaluation strategy that uses mean squared error
    '''
    def calcualte_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("calcualting MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE : {}".format(mse))
            return mse
        except Exception as e:
            logging.error("error in calculating MSE")
            raise e
        
class RMSE(Evaluation):
    '''
    Evaluation strategy that uses root mean squared error
    '''
    def calcualte_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info("RMSE : {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("error in calcualating RMSE")
            raise e

class R2(Evaluation):
    '''
    Evaluation strategy that uses R2 score
    '''
    def calcualte_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 : {}".format(r2))
            return r2
        except Exception as e:
            logging.error("error in calcualating R2 score")
            raise e