import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                # "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0)
            }

            params = {
                "LinearRegression": {},  # no hyperparameters usually tuned here

                "DecisionTreeRegressor": {
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },

                "RandomForestRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                },

                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_features": ["sqrt", "log2", None]
                },

                # "XGBRegressor": {
                #     "n_estimators": [100, 200],
                #     "learning_rate": [0.01, 0.05, 0.1],
                #     "max_depth": [3, 5, 7],
                #     "subsample": [0.6, 0.8, 1.0],
                #     "colsample_bytree": [0.6, 0.8, 1.0]
                # },

                "KNeighborsRegressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]  # 1 = Manhattan, 2 = Euclidean distance
                },

                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "loss": ["linear", "square", "exponential"]
                },

                "CatBoostRegressor": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7],
                    "border_count": [32, 50, 100]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,params=params)
            
            # Sort the model report based on the score
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
            
        
