from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy
import numpy as np


def create_xgboost(n_estimators: int, max_depth: int, learning_rate: float, subsample: float)-> XGBRegressor:
    """
    create the xgboost regressor model with hyperparameters
    
    Args:
        n_estimators (int):
        
        max_depth (int):
        
        learning_rate (float):
        
        subsample (float):
        
        
    Returns:
        xgb (XGBRegressor):
            model init with hyperparameters

    """
    xgb = XGBRegressor(n_estimators= n_estimators, 
                       max_depth=max_depth, 
                       learning_rate=learning_rate, 
                       subsample=subsample)
    
    return xgb


def create_rfr(n_estimators: int, max_depth: int, min_samples_split: int, min_samples_leaf: int)-> RandomForestRegressor:
    """
        Args:
            n_estimators (int):
            
            max_depth (int):
            
            min_samples_split (int):

            min_samples_leaf (int):
        
        Returns:
            rfr (RandomForestRegressor):
                model init with hyperparameters
    """
    rfr = RandomForestRegressor(n_estimators= n_estimators, 
                                max_depth= max_depth, 
                                min_samples_split= min_samples_split, 
                                min_samples_leaf= min_samples_leaf)
    
    return rfr


def preprocess_data(data: pd.DataFrame)-> numpy.array:
    """ 
        Function to preprocess the dataset before training model
        
        Args:
            data (pd.DataFrame):
                the pandas dataframe we want to preprocess
                
        Returns:
            x_train (numpy.array):
                training dataset
                
            x_val (numpy.array):
                validation dataset
                
            y_train (numpy.array): 
                label for training dataset
                
            y_val (numpy.array):
                label for validation dataset
                
        Raise:
        ------
            - ValueType Error
            - if data is not pd.DataFrame
    
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"wrong type for data, expected pd.DataFrame got {type(data).__name__}")
    
    column_non_tabular = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
    
    encoder = LabelEncoder()

    for column in column_non_tabular:
        data[column] = encoder.fit_transform(data[column])
        print(column)
        labels = encoder.classes_
        encoded_value_mapping = dict(enumerate(labels))
        print("Mapping des valeurs encodées et des labels :", encoded_value_mapping)
        
    # add a column for total room and mean dimension size room

    data["total_room"]          = data["bedrooms"] + data["bathrooms"] + data["guestroom"] + data["basement"]
    data["mean_dimension_room"] = data["area"] / data["total_room"]
    data["mean_dimension_room"] = data["mean_dimension_room"].apply(lambda x: round(x))
    
    # X and y
    price = data["price"]
    train = data.drop(["price"], axis= 1)

    # Normalize, no need here
    norm = MinMaxScaler()
    #train = norm.fit_transform(train)

    # split
    x_train, x_val, y_train, y_val = train_test_split(train, price, test_size = 0.3, random_state = 42)
    
    return x_train, x_val, y_train, y_val

def create_query_dataframe(query: dict):
    
    data_query = pd.DataFrame([query])
    data_query["total_room"]          = data_query["bedrooms"] + data_query["bathrooms"] + data_query["guestroom"] + data_query["basement"]
    data_query["mean_dimension_room"] = data_query["area"] / data_query["total_room"]
    data_query["mean_dimension_room"] = data_query["mean_dimension_room"].apply(lambda x: round(x))
    
    return data_query
    
    


def train_and_save_xgboost(x_train: pd.DataFrame, x_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series)->XGBRegressor:
    """
        train and then save the xgboost model

        Args:
            x_train (numpy.array):
                    training dataset
                    
            x_val (numpy.array):
                validation dataset
                
            y_train (numpy.array): 
                label for training dataset
                
            y_val (numpy.array):
                label for validation dataset

        Returns:
            XGBRegressor: 
                trained model
        
        Raise:
        ------
            - ValueType Error
            - if input are not numpy.array
    """
    
    if not isinstance(x_train, pd.DataFrame) or not isinstance(x_val, pd.DataFrame) or not isinstance(y_train, pd.Series) or not isinstance(y_val, pd.Series):
        raise TypeError(f"Wrong type for data, expected pd.DataFrame or pd.Series got x_train: {type(x_train).__name__}, "
                        f"x_val: {type(x_val).__name__}, y_train: {type(y_train).__name__}, "
                        f"y_val: {type(y_val).__name__}")
    xgb = create_xgboost(n_estimators = 100, 
                        max_depth = 6, 
                        learning_rate = 0.1, 
                        subsample = 0.8) 

    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_val)
    score = r2_score(y_val, pred)
    
    return xgb

def train_and_save_randomforest(x_train: numpy.array, x_val: numpy.array, y_train: numpy.array, y_val: numpy.array)->RandomForestRegressor:
    """
        train and then save the randomforest model

        Args:
            x_train (numpy.array):
                    training dataset
                    
            x_val (numpy.array):
                validation dataset
                
            y_train (numpy.array): 
                label for training dataset
                
            y_val (numpy.array):
                label for validation dataset

        Returns:
            XGBRegressor: 
                trained model
        
        Raise:
        ------
            - ValueType Error
            - if input are not numpy.array
    """
    
    # if not isinstance(x_train, numpy.array) or not isinstance(x_val, numpy.array) or not isinstance(y_train, numpy.array) or not isinstance(y_val, numpy.array):
    #     raise TypeError(f"wrong type for data, expected numpy.array got x_train: {type(x_train).__name__}, x_val: {type(x_val).__name__}, y_train: {type(y_train).__name__}, y_val :{type(y_val).__name__}")

    rfr = create_rfr(n_estimators = 100, 
                    max_depth = 6, 
                    min_samples_split = 3, 
                    min_samples_leaf = 10) 

    rfr.fit(x_train, y_train)

    pred = rfr.predict(x_val)
    score = r2_score(y_val, pred)
    
    return rfr

def make_prediction(data: numpy.array, model: RandomForestRegressor | XGBRegressor)-> float:
    """
        Make prediction with custom pretrained model

        Args:
            data (numpy.array): 
                the array of value we need to predict the label
                
            model (RandomForestRegressor | XGBRegressor): 
                the pretrained model 

        Returns:
            result (float): 
                the predicted value

        
    """
    result = model.predict(data)
    
    return result