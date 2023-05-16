import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import save_obj

@dataclass
class Data_transformation_config:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=Data_transformation_config()

    def get_preprocessor_obj(self):
        try:
            
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            numrical_pipeline=Pipeline(
                steps=[
                        ("imputer",SimpleImputer(strategy='median')),
                        ("standardscaling",StandardScaler())
                ]

            )

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoding",OneHotEncoder()),
                    ("standardscaling",StandardScaler(with_mean=False)),

                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("numeric_pipeline",numrical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
            logging.info('Numerical & Categorical columns are transformed')

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            train_data.columns=train_data.columns.str.replace(" ","_").str.replace("/","_")
            test_data.columns=test_data.columns.str.replace(" ","_").str.replace("/","_")
            logging.info('Read train & test data and removed space from column names')

            target_column='math_score'

            train_input=train_data.drop(columns=[target_column],axis=1)
            train_target=train_data[target_column]

            test_input=test_data.drop(columns=[target_column],axis=1)
            test_target=test_data[target_column]

            logging.info('split dependent & independent features')

            preprocessing_obj=self.get_preprocessor_obj()
            logging.info('created preprocessing object')

            train_input_arr=preprocessing_obj.fit_transform(train_data)
            test_input_arr=preprocessing_obj.fit_transform(test_data)

            train_arr=np.c_[train_input_arr,np.array(train_target)]
            test_arr=np.c_[test_input_arr,np.array(test_target)]

            save_obj(fil_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

