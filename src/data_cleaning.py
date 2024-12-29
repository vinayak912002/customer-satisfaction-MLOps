'''
here we are using strategy pattern in which we first describe an abstract class
then we write classes for different strategies
then we write the final class which makes use of these strategies
'''

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handling data.
    '''

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    '''
    Strategy to preprocess data
    '''

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        preprocess data
        '''
        try:
            '''
            for simplicity and for learning MLOps we are keeping the preprocessing simple
            '''
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
            
            axis=1
            )

            data["product_weight_g"] = data["product_weight_g"].fillna(data["product_weight_g"].median())
            data["product_length_cm"] = data["product_length_cm"].fillna(data["product_length_cm"].median())
            data["product_height_cm"] = data["product_height_cm"].fillna(data["product_height_cm"].median())
            data["product_width_cm"] = data["product_width_cm"].fillna(data["product_width_cm"].median())
            data["review_comment_message"] = data["review_comment_message"].fillna("no review")

            data = data.select_dtypes(include=["number"])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("error in preprocessing data : {}".format(e))
            raise e
        
class DataDivideStrategy(DataStrategy):
    '''
    Strategy for dividing data into train and test
    '''

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        '''
        divide data intop train and test
        '''
        try:
            X = data.drop(['review_score'], axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return  X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("error in dividing data: {}".format(e))
            raise e
        
class DataCleaning:
    '''
    Class which processes the data and divides it into train and test
    '''
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self)-> pd.DataFrame | pd.Series:
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("error in cleaning data: {}".format(e))       