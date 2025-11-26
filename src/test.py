
import pandas as pd
import glob, joblib, warnings

from datetime import datetime 
from pyprojroot import here
from sklearn.metrics import mean_absolute_error

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
pd.reset_option("mode.chained_assignment")

def test(df:pd.DataFrame, sex:str):

    datem = datetime.today().strftime("%Y-%m-%d")
    if sex=='male':
        for file in glob.glob(f"{here()}/models/????-??-??_male_xgboost.joblib.dat"):
            model = joblib.load(file)
        for file in glob.glob(f"{here()}/models/????-??-??_male_alpha_beta.pkl"):
            params = pd.read_pickle(file)

    elif sex=='female':
        for file in glob.glob(f"{here()}/models/????-??-??_female_xgboost.joblib.dat"):
            model = joblib.load(file)
        for file in glob.glob(f"{here()}/models/????-??-??_female_alpha_beta.pkl"):
            params = pd.read_pickle(file)

    else:
        for file in glob.glob(f"{here()}/models/????-??-??_all_test_xgboost.joblib.dat"):
            model = joblib.load(file)
        for file in glob.glob(f"{here()}/models/????-??-??_all_alpha_beta.pkl"):
            params = pd.read_pickle(file)

    try:
        df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)
    except:
        print("Unnamed: 0 columns dont exist")

    if 'chronological_age' in df.columns:
        pass
    else:
        df.rename(columns={'scan_age': 'chronological_age'}, inplace=True)

    scan_age_df = df['chronological_age']

    df = df.drop(['chronological_age', 'scan_id'], axis=1)

    print(df.shape)

    predictions = model.predict(df)
    mae = mean_absolute_error(scan_age_df, predictions)
    print(f"MAE: {mae}")

    # Bias correction
    df_result = pd.DataFrame({"chronological_age":scan_age_df, "brain_age":predictions},)
    df_result["brain_age_gap"] = df_result["brain_age"] - df_result["chronological_age"]

    #Vlad's method
    df_result['adjusted_brain_age'] = (df_result['brain_age'] - params['alpha']) / params['beta']
    df_result["adjusted_brain_age_gap"] = df_result["adjusted_brain_age"] - df_result["chronological_age"]

    mae_adj = mean_absolute_error(df_result['chronological_age'], df_result['adjusted_brain_age'])
    print(f"Adjusted MAE: {mae_adj}")
    
    result = df_result

    age_brain_age_corr = result["chronological_age"].corr(result["brain_age"])
    age_adj_brain_age_corr = result["chronological_age"].corr(result["adjusted_brain_age"])

    age_bag_corr = result["chronological_age"].corr(result["brain_age_gap"])
    age_adjbag_corr = result["chronological_age"].corr(result["adjusted_brain_age_gap"])
    

    print(f"Correlation (Age * brain_age): {age_brain_age_corr}")
    print(f"Correlation (Age * adjusted_brain_age): {age_adj_brain_age_corr}")

    print(f"Correlation (Age * brain_age_gap): {age_bag_corr}")
    print(f"Correlation (Age * adjusted_brain_age_gap): {age_adjbag_corr}")
    

    return result
    

if __name__ == '__main__':

    test()
