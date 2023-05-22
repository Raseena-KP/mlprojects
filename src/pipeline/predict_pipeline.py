from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from src.utils import load_obj
import sys


class Predict_pipeline:
    def __init__(self):
        pass
    
    def predict(sefl,features):

        try:
            pre_processor=load_obj(fil_path='artifacts/preprocessor.pkl')
            model_obj=load_obj(fil_path='artifacts/model.pkl')
            preprocessed_data=pre_processor.transform(features)
            result=model_obj.predict(preprocessed_data)
            return result
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                gender:str,
                race_ethnicity:str,
                parental_level_of_education:str,
                lunch:str,
                test_preparation_course:str,
                reading_score:float,
                writing_score:float
                ):

        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def convert_input_to_dataframe(self):
        try:
            input_dict={
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score],
            }
            return pd.DataFrame(input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)

        