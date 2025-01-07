import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            
            numerical_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy="median")),
                         ("scaler",StandardScaler())]
            )

            categorical_pipeline = Pipeline(
                steps = [("imputer",SimpleImputer(strategy="most_frequent")),
                         ("one_hot_encoder",OneHotEncoder()),
                         ("scaler",StandardScaler(with_mean=False))]
            )

            logging.info(f"Categorical features: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [("numerical_pipeline",numerical_pipeline,numerical_features),
                 ("categorical_pipeline",categorical_pipeline,categorical_features)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read the training and testing data')
            
            logging.info('Get the preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()
            
            target_feature = 'math_score'
            numerical_features = ['writing_score','reading_score']
            
            input_features_train_df = train_df.drop(columns = [target_feature],axis = 1)
            target_feature_train_df = train_df[target_feature]
            
            input_features_test_df = test_df.drop(columns = [target_feature],axis = 1)
            target_feature_test_df = test_df[target_feature]

            logging.info('Applying the preprocessing object on the training and testing dataframe')

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df) 
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df) 

            train_arr = np.c_[input_features_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            logging.info('Save the preprocessing object')

            save_object(
            file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
            )

            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
    

    







