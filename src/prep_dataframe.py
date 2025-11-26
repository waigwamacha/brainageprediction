
import argparse
import pandas as pd
import pickle
from pyprojroot import here

def read_dataframe(filename):
    
    df = filename 
    vlad_features = pd.read_csv(f"{here()}/references/all_features.csv")
    features = vlad_features.feature_names.tolist()
    
    df.columns = [x.lower() for x in df.columns]
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    features = [item.lower() for item in features]
    
    features.append('scan_id')
    features.append('scan_age')
    features.append('scan_time')
    df = df[df.columns[df.columns.isin(features)]].copy()
    df = df.reindex(sorted(df.columns), axis=1)
    print(f"{df.shape}")

    return df

def prep_sex_train_data(filename):
    
    df = pd.read_csv(filename,index_col=False)
    vlad_features = pd.read_csv(f"{here()}/references/all_features.csv")
    features = vlad_features.feature_names.tolist()

    df.columns = [x.lower() for x in df.columns]
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    features = [item.lower() for item in features]
    

    features.append('scan_id')
    df = df[df.columns[df.columns.isin(features)]]
    df = df.reindex(sorted(df.columns), axis=1)

    df_male = df[df['sex'] == 'Male']
    df_female = df[df['sex'] == 'Female']

    Y_male = pd.DataFrame(df_male['scan_age'])
    X_male = df_male.drop(['sex', 'scan_age'], axis=1)

    Y_female = pd.DataFrame(df_female['scan_age'])
    X_female = df_female.drop(['sex', 'scan_age'], axis=1)

    X_train_male = X_male.to_numpy() 
    Y_train_male = Y_male.to_numpy()

    X_train_female = X_female.to_numpy() 
    Y_train_female = Y_female.to_numpy() 

    with open(f'{here()}/data/train/x_train_male.pkl','wb') as f:
        pickle.dump(X_train_male, f)

    with open(f'{here()}/data/train/y_train_male.pkl','wb') as f:
        pickle.dump(Y_train_male, f)

    with open(f'{here()}/data/train/x_train_female.pkl','wb') as f:
        pickle.dump(X_train_female, f)

    with open(f'{here()}/data/train/y_train_female.pkl','wb') as f:
        pickle.dump(Y_train_female, f)

    return X_male, Y_male, X_female, Y_female

if __name__ == '__main__':
    
    read_dataframe()
