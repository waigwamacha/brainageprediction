
import glob
import os
import sys 

import pandas as pd
from pyprojroot import here
sys.path.insert(0, f"{here()}/src")
from prep_columns import clean_columns
from prep_dataframe import read_dataframe

def test_raw_cleaned_dataframe_shape():

    data_directory = f"{here()}/data"
    file_paths = glob.glob(os.path.join(f'{data_directory}', '**', '*_aparc_merged*'), recursive=True)
    
    for path in file_paths:
        filename = pd.read_csv(path, sep='\t')
        
        df = clean_columns(filename)
        assert df.shape[1] == 188


def test_processed_test_df_shape():

    data_directory = f"{here()}/data"

    files = [f'{data_directory}/processed/frb_test_20240503.csv']

    for path in files:
        file = pd.read_csv(path)
        df = read_dataframe(file)
        assert df.shape[1] == 188


def test_train_data_shape():

    data_directory = f"{here()}/data/raw/train"
    file_paths = glob.glob(os.path.join(f'{data_directory}', '**', '*_train*'), recursive=True)

    for path in file_paths:
        if path.endswith('.pkl'):
            filename = pd.read_pickle(path)
            assert filename.shape[1] == 187
        else:
            filename = pd.read_csv(path)
            assert filename.shape[1] == 187




    

