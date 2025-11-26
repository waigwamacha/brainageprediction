# Scripts

- **prep_dataframe.py**: script to prepare dataframes

  - To read dataframes & get them into shape for training or testing:

    ```{python}
    from prep_dataframe import read_dataframe
    df = read_dataframe(some_dataframe)
    ```

  - To prep sex data for training:

    ```{python}
    from prep_dataframe import prep_sex_train_data
    prep_sex_train_data(df_with_sex_column)
    ```

- **hpo.py**: script to run hyperparameter optimization using optuna. To run in notebook and get best hyperparameters:

  - mlflow

    ```{python}
    mlflow server --backend-store-uri sqlite:///backend.db
    ```

    ```{python}
    from hpo import hyperparameter_tuning
    hyperparameter_tuning("female")
    ```

    or to run unattended:

    ```{python}
    nohup python3 src/hpo.py --sex 'female' > /home/murage/Desktop/neptune/logs/hop_male.txt 2>&1 &
    ```

- **train.py**: script to train xgboost models (male, female, all) using best hyperparameters obtained using bayesian optimization in optuna. example:

    ```{python}
    from train import train
    train("female")
    ```

    ```{python}
    python3 src/train.py --sex "female"
    ```

- **bias_correct.py**: script to do bias correction (male, female, all) using the training set. example:

    ```{python}
    python3 src/bias_correct.py --sex "female"
    ```

    ```{python}
    from bias_correct import bias_correction
    bias_correction("female")
    ```

- **test.py** script to predict on test data (NB: Can't be run from the terminal yet since we need to input a dataframe into the function)

    ```{python}
    from test import test
    results = test(some_dataframe, "female")
    ```
