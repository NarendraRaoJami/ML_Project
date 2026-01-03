import os
import sys

import numpy as np
import pandas as pd

import dill 

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            params = param.get(model_name, {})

            if model.__class__.__name__ == "CatBoostRegressor":
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                report[model_name] = r2_score(y_test, y_test_pred)
                continue

            if params:
                gs = GridSearchCV(
                    model,
                    params,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
