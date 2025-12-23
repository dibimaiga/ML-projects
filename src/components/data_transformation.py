#We have numerical and categorical variables, What kind of transformation we'll apply to each
#The main purpose of data transformation is to do basically Feature Engineering, data cleaning, convert my 
#categorical features into numerical

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline #for implementing the pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

# Let's create inputs required for the data transformation path 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","prepocessor.pkl")

class DataTransformation:
    def __init__(self):
        """
        This function is responsible for data transformation based on the different types of data

        :param self: Description
        """
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
            ]

            #let's create pipelines for both

            num_pipeline = Pipeline( # numerical pipeline
                steps= [
                    ("imputer",SimpleImputer(strategy="median")), #handling missing values
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline( # categorical pipeline
                steps= [
                    ("imputer",SimpleImputer(strategy="most_frequent")), #handling missing values
                    ("one_hot_encoder",OneHotEncoder()), #one_hot_encoding
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
# Let's just display our numerical and categorical columns
            logging.info(f"categorical columns: {categorical_columns}")
            logging.info(f"numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer( # combining the two pipelines
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor #we're returning this entire thing
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            #Now i'll be reading all my preprocessing object

            preprocessor_obj = self.get_data_transformer_object() #preprocessor_obj whatever object i created on top (this one : return preprocessor)
            #I will be getting this particular object over here

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            print(input_feature_train_df)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and on test dataframe ")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
# we need to convert our preprocessor into   pkl file. We've already taken a path and we're going to save this pkl file into the same same location

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
#So above with rhis function , we're saving the pkl file in the hard disk
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        #if something happened succesfully but you don't see error , check if you write the exception
        






