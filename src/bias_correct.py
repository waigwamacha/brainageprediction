
import argparse, joblib, glob,sys
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

from datetime import datetime
from pyprojroot import here
from scipy import stats

sys.path.insert(0, f"{here()}/src")
from plots import scatter_bag_age, scatter_adjbag_age

def bias_correction(sex:str):
    
    datem = datetime.today().strftime("%Y-%m-%d")

    if sex=='male':
        X_train = pd.read_pickle(f"{here()}/data/train/x_train_male_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_male_ext.pkl")

        for file in glob.glob(f"{here()}/models/????-??-??_male_xgboost.joblib.dat"):
            model = joblib.load(file)

    elif sex=='female':
        X_train = pd.read_pickle(f"{here()}/data/train/x_train_female_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_female_ext.pkl")
        
        for file in glob.glob(f"{here()}/models/????-??-??_female_xgboost.joblib.dat"):
            model = joblib.load(file)

    else:
        X_train = pd.read_pickle(f"{here()}/data/train/x_train_ext.pkl")
        Y_train = pd.read_pickle(f"{here()}/data/train/y_train_ext.pkl")

        for file in glob.glob(f"{here()}/models/????-??-??_all_test_xgboost.joblib.dat"):
            model = joblib.load(file)

    print(X_train.shape, Y_train.shape)

    y_pred = model.predict(X_train)

    y_train = np.reshape(Y_train, -1)
    y_test = pd.DataFrame(y_train)
    y_test.rename(columns={0: "chronological_age"}, inplace=True)
    y_pred = pd.DataFrame(y_pred)
    y_pred.rename(columns={0: "brain_age"}, inplace=True)

    results = pd.concat([y_test, y_pred], axis=1)
    results['brain_age_gap'] = results['brain_age'] - results['chronological_age']

    r, p = stats.pearsonr(results["chronological_age"], results["brain_age_gap"]) 
    print(f"r: {r}, p-value: {p:.4f}")

    scatter_bag_age(results, sex)

    """ x = results["chronological_age"]
    y = results["brain_age_gap"] 
    X = sm.add_constant(x)

    ols = sm.OLS(y, X).fit()
    alpha = ols.params[0]
    beta = ols.params[1]
    alpha_beta = {}
    alpha_beta['alpha'] = alpha
    alpha_beta['beta'] = beta """

    #Vlad's bias correction method (Cole et al., 2019)
    x = results["chronological_age"]
    y = results["brain_age"] 
    X = sm.add_constant(x)

    ols = sm.OLS(y, X).fit()
    alpha = ols.params[0]
    beta = ols.params[1]
    alpha_beta = {}
    alpha_beta['alpha'] = alpha
    alpha_beta['beta'] = beta

    if glob.glob(f"{here()}/models/????-??-??_{sex}_test_alpha_beta.pkl"):
        pass
    else:
        with open(f'{here()}/models/{datem}_{sex}_test_alpha_beta.pkl', 'wb') as f:
            pickle.dump(alpha_beta, f)

    results['adjusted_brain_age'] = (results['brain_age'] - ols.params[0]) / ols.params[1]
    results["adjusted_brain_age_gap"] = results["adjusted_brain_age"] - results["chronological_age"]
    #MCsweeney et al., 2024
    #results['corrected_brain_age'] = (results['brain_age'] + (results["chronological_age"] - (ols.params[0]*results["chronological_age"]*ols.params[1])))
    #results["corrected_brain_age_gap"] = results["corrected_brain_age"] - results["chronological_age"]
    scatter_adjbag_age(results)

    bag_age_mean = results['brain_age_gap'].corr(results['chronological_age'])
    adjbag_age_mean = results['adjusted_brain_age_gap'].corr(results['chronological_age'])
    adjbag_brainage_mean = results['adjusted_brain_age_gap'].corr(results['brain_age'])
    adjbag_bag_mean = results['adjusted_brain_age_gap'].corr(results['brain_age_gap'])
    
    print(f"BAG-Age: {bag_age_mean:.4f}\n")
    print(f"AdjBAG-Age: {adjbag_age_mean:.4f}\n")
    print(f"AdjBAG-BrainAge: {adjbag_brainage_mean:.4f}\n")
    print(f"AdjBAG-BAG: {adjbag_bag_mean:.4f}\n")

    #print(f"AdjBAG-Age\n{results['adjusted_brain_age_gap']}.corr({results['chronological_age']})")
    #print(f"\nAdjBAG-BrainAge\n{results['adjusted_brain_age_gap']}.corr({results['brain_age']})")
    #print(f"\nAdjBAG-BAG\n{results['adjusted_brain_age_gap']}.corr({results['brain_age_gap']})")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sex",
        help="sex: male, female, all"
    )
    
    args = parser.parse_args()
    bias_correction(args.sex)