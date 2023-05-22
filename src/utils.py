import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill # another library to create pickle file
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

def save_obj(fil_path,obj):
    try:
        dir_path=os.path.dirname(fil_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(fil_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):

    try:
        model_perform={}

        for model_name, model in models.items():
            #train the model
            gv=GridSearchCV(model,param_grid=param[model_name],cv=3)
            gv.fit(x_train,y_train)
            
            model.set_params(**gv.best_params_)
            model.fit(x_train,y_train)

            #make prediction
            y_pred=model.predict(x_test)

            # calculate accuracy score
            r2_score_val=r2_score(y_test,y_pred)

            model_perform[model_name]=r2_score_val

        return model_perform

    except Exception as e:
        raise CustomException(e,sys)

def load_obj(fil_path):
    try:
        with open(fil_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        CustomException(e,sys)