#This is the most important part
#We'le create a web application that will take a data as input and in the backend all the prepocessing and traing 
# will operate and the app will output a prediction of the student performance

import sys #exception
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline():
    def __init__(self):
        pass

    def predict(self,features): # what is basically doing my prediction

        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','prepocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData(): # will be responsible in mapping all the inputs that we're giving in the html to the
    #backend with this particular values

    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        #Above we've just listed the features that we needed

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

#I will create a function that will return all my inputs as a dataframe because we trained our models in a form dataframe 
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)
        #here whatever inputs that will be given from the web application, those same inputs will be mapped with these particular values

        except Exception as e:
            raise CustomException(e, sys)


