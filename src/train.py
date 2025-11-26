from datetime import datetime
from pyprojroot import here

import argparse, glob, joblib, pickle, re, warnings, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import xgboost as xgb
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from plots import validation_error_plot, feature_importance_plot
from plotnine import *

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train(sex:str):

    datem = datetime.today().strftime("%Y-%m-%d")
    vlad_features = pd.read_csv(f"{here()}/references/all_features.csv")
    feature_names = vlad_features.feature_names.tolist()

    if sex=='male':
    
        X_train = pd.read_pickle(f"{here()}/data/train/x_train_male_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_male_ext.pkl")

        X_val = pd.read_pickle(f"{here()}/data/train/x_val_male_ext.pkl")
        Y_val = pd.read_pickle(f"{here()}/data/train/y_val_male_ext.pkl")

        for file in glob.glob(f"{here()}/models/params/????-??-??_male_best_params_ext.pkl"):
            with open(file, "rb") as f_in:
                params = pickle.load(f_in)

    elif sex=='female':

        X_train = pd.read_pickle(f"{here()}/data/train/x_train_female_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_female_ext.pkl")

        X_val = pd.read_pickle(f"{here()}/data/train/x_val_female_ext.pkl")
        Y_val = pd.read_pickle(f"{here()}/data/train/y_val_female_ext.pkl")
        
        for file in glob.glob(f"{here()}/models/params/????-??-??_female_best_params_ext.pkl"):
            with open(file, "rb") as f_in:
                params = pickle.load(f_in)

    else:

        X_train = pd.read_pickle(f"{here()}/data/train/x_train_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_ext.pkl")

        X_val = pd.read_pickle(f"{here()}/data/train/x_val_ext.pkl")
        Y_val = pd.read_pickle(f"{here()}/data/train/y_val_ext.pkl")
        
        for file in glob.glob(f"{here()}/models/params/????-??-??_all_best_params_ext.pkl"):
            with open(file, "rb") as f_in:
                params = pickle.load(f_in)


    X_train_male = pd.read_pickle(f"{here()}/data/train/x_train_male_ext.pkl")
    Y_train_male = pd.read_pickle(f"{here()}/data/train/y_train_male_ext.pkl")

    X_train_fem = pd.read_pickle(f"{here()}/data/train/x_train_female_ext.pkl")
    Y_train_fem = pd.read_pickle(f"{here()}/data/train/y_train_female_ext.pkl")

    X_train_all = pd.read_pickle(f"{here()}/data/train/x_train_ext.pkl")
    Y_train_all = pd.read_pickle(f"{here()}/data/train/y_train_ext.pkl")
    
    model = xgb.XGBRegressor(**params, eval_metric="mae")

    reg = model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_val, Y_val)], verbose = 0)

    validation_error_plot(reg, params['learning_rate'])

    model.fit(X_train, Y_train, verbose=0)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(Y_val, y_pred)
    print(f'Validation Mean Absolute Error: {mae:.2f}')

    male_pred = model.predict(X_train_male)
    mae = mean_absolute_error(Y_train_male, male_pred)
    print(f'Male Mean Absolute Error: {mae:.2f}')

    female_pred = model.predict(X_train_fem)
    mae = mean_absolute_error(Y_train_fem, female_pred)
    print(f'Female Mean Absolute Error: {mae:.2f}')

    all_pred = model.predict(X_train_all)
    mae = mean_absolute_error(Y_train_all, all_pred)
    print(f'All participants Mean Absolute Error: {mae:.2f}')

    feature_importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importances))

    cols = ["feature_importance"]
    df = pd.DataFrame.from_dict(feature_importance_dict, orient='index')
    df.columns = cols
    df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
    df.index.names = ['feature_name']
    df = df.reset_index()

    res = df.head(20)
    #df.to_csv(f"{here()}/data/interim/{sex}_feature_importances.csv", index=False)

    name_mapping = {
    'FS_L_Vessel_Vol': 'Left Vessel Volume',
    'FS_R_Lateralorbitofrontal_GrayVol': 'Right Lateralorbitofrontal GrayVol',
    'FS_L_Superiortemporal_GrayVol': 'Left Superiortemporal GrayVol',
    'FS_L_Transversetemporal_Area': 'Left Transversetemporal Area',
    'FS_R_Transversetemporal_GrayVol': 'Right Transversetemporal GrayVol',
    'FS_TotCort_GM_Vol': 'Total cortical gray matter volume',
    'FS_R_Parahippocampal_GrayVol': 'Right Parahippocampal GrayVol',
    'FS_R_Frontalpole_Area': 'Right Frontalpole Area',
    'FS_R_Isthmuscingulate_Area':'Right Isthmuscingulate Surface Area',
    'FS_R_Rostralanteriorcingulate_GrayVol': 'Right Rostralanteriorcingulate GrayVol',
    #'FS_R_Medialorbitofrontal_GrayVol': 'Right Medialorbitofrontal GrayVol',
    #'FS_L_Superiortemporal_Area': 'Left Superiortemporal Area',
    #'FS_R_Superiorparietal_GrayVol': 'Right Superiorparietal GrayVol',
    #'FS_L_Precuneus_Area': 'Left Precuneus_Area',
    #'FS_RCort_GM_Vol': 'Right hemisphere cortical gray matter volume',
    }
    res = res.copy()
    res['feature_name'] = res['feature_name'].replace(name_mapping)

    res['feature_name'] = pd.Categorical(res['feature_name'], categories=res['feature_name'][::-1], ordered=True)

    bar_plot = feature_importance_plot(res)
    bar_plot.save(f"{here()}/figures/{sex}_feature_importance_plot.png", dpi=800)

    file_path = f"{here()}/models/????-??-??_all_second_xgboost.joblib.dat"

    #for file in glob.glob(f"{here()}/models/????-??-??_all_second_xgboost.joblib.dat"):
    if os.path.exists(file_path):
        pass
    else:
        joblib.dump(model, f"{here()}/models/{datem}_{sex}_test_xgboost.joblib.dat")
                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sex",
        help="sex: male, female, all"
    )
        
    args = parser.parse_args()

    train(args.sex)







