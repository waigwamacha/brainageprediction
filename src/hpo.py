
import argparse
#import mlflow
import optuna

import numpy as np
import pandas as pd
import os, pickle, warnings
import xgboost as xgb

from datetime import datetime
from optuna import trial
from pyprojroot import here
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RepeatedKFold

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment("xgboost optimization")

warnings.simplefilter(action='ignore', category=FutureWarning)

def hyperparameter_tuning(sex: str):

    datem = datetime.today().strftime("%Y-%m-%d")

    if sex=='male':

        X_train = pd.read_pickle(f"{here()}/data/train/x_train_male_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_male_ext.pkl")

    elif sex=='female':

        X_train = pd.read_pickle(f"{here()}/data/train/x_train_female_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_female_ext.pkl")

    else:

        X_train = pd.read_pickle(f"{here()}/data/train/x_train_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_ext.pkl")

    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, shuffle=True, random_state=42)
    
    print(X_train.shape, Y_train.shape)
    print(f"{x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")

    def objective(trial):
        #Great piece on hpo in xgboost: https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/
        params = {
        'objective': 'reg:absoluteerror',
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1500),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 36),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        }

        # Fit the model
        #with mlflow.start_run():
        #    mlflow.log_params(params)
        optuna_model = xgb.XGBRegressor(**params)

        rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)

        mae_scores = []

        # Perform cross-validation
        for train_index, test_index in rkf.split(x_train):        
            optuna_model.fit(x_train, y_train)

            # Make predictions
            y_pred = optuna_model.predict(x_test)

            # Evaluate predictions
            mae = mean_absolute_error(y_test, y_pred)
            #mlflow.log_metric("mae", mae)
            mae_scores.append(mae)

        #trial.set_user_attr('mae', mae)
        return np.mean(mae_scores)

    study = optuna.create_study(**{
    'study_name': f'brainAge_{sex}', 
    'direction':'minimize'
    })

    study.optimize(**{
        'func': objective, 
        'n_trials': 50, 
        'show_progress_bar': False
        })
    
    print(f"{study.best_params}")
    print("Best trial:")
    trial = study.best_trial
    print("Value: {:.4f}".format(trial.value))
    print("Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    with open(f"{here()}/models/params/{datem}_{sex}_second_best_params_ext.pkl", 'wb') as outfile:
        pickle.dump(study.best_params, outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sex",
        help="sex: male, female, all"
    )
    
    args = parser.parse_args()
    hyperparameter_tuning(args.sex)
    