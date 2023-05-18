import os
import sys

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor

from dataclasses import dataclass
from src.utils import evaluate_models,save_obj

@dataclass
class ModelTrainerConfig:
    model_trainer_config_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_data,test_data):
        try:
            x_train=train_data[:,:-1]
            y_train=train_data[:,-1]

            x_test=test_data[:,:-1]
            y_test=test_data[:,-1]
            
            models={
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }

            params = {
                 "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ridge":{"alpha": [0.1, 0.5, 1.0]},
                "Lasso":{"alpha": [0.1, 0.5, 1.0]},
                "KNeighborsRegressor":{"n_neighbors": [3, 5, 7]}
                
            }

            model_performance:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

            Best_Model_score = max(model_performance)
            Best_Model=max(model_performance, key=model_performance.get)

            model_obj=models[Best_Model]


            save_obj(fil_path=self.modeltrainer_config.model_trainer_config_path,obj=model_obj)

            y_pred=model_obj.predict(x_test)

            r2_score_value=r2_score(y_test,y_pred)

            logging.info('Obtained Best model for train & test data')

            return r2_score_value

            
        except Exception as e:
            raise CustomException(e,sys)

