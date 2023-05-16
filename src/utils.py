import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill # another library to create pickle file

def save_obj(fil_path,obj):
    try:
        dir_path=os.path.dirname(fil_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(fil_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        CustomException(e,sys)