# to learn about the strategy pattern used in this file go to the data_cleaning.py file
import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    '''
    abstract class for all models
    '''

    @abstractmethod
    def train(self, X_train, y_train):
        '''
        trains the model
        '''
        pass
    
    
class LinearRegressionModel(Model):
    '''
    linear regression model
    '''
    def train(self, X_train, y_train, **kwargs):
        '''
        trains the model

        Args:
            X_train: Training data
            y_train: Training labels

        returns:
            None
        '''
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model : {}".format(e))
            raise e