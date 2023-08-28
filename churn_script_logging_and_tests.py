'''
Script with test functions to be run with `pytest` to test 'churn_library.py'

Author: Dallas Lea
August 2023
'''

import os
import logging
import pytest
import churn_library as cls


@pytest.fixture(scope='module')
def df():
    '''
    DataFrame object fixture to use in eda and encoder test functions
    '''
    df = cls.import_data("./data/bank_data.csv")
    return df


def cleanup_dir(directory):
    '''
    Removes all files in a directory
    input:
        directory: Directory path
    output:
        None
    '''
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    print("Running test_import function")
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        df.head()
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    print("Running test_import function")


def test_eda(df):
    '''
    test perform_eda function
    '''
    image_dir_eda = './images_test/eda/'
    cleanup_dir(image_dir_eda)
    print("Running test_eda function")
    cls.perform_eda(df, image_dir_eda=image_dir_eda)
    try:
        assert os.path.isfile(image_dir_eda + "corrplot.png")
        logging.info("Correlation plot created")
    except AssertionError as err:
        logging.error("Age history plot not created.")


@pytest.fixture(scope='module')
def df2(df):
    '''
    DataFrame object fixture, encoded to use in feature engineering
    and model training test functions
    '''
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    df2 = cls.encoder_helper(df, cat_columns)
    try:
        #Use 'or' to check for existence of either one-hot encoded gender column,
        #just in case bank_data.csv has only one gender
        assert ('Gender_F' in df2.columns or 'Gender_M' in df2.columns)
        logging.info("One or more one-hot encoded columns created for Gender")
    except AssertionError as err:
        logging.error("No one-hot encoded columns created for Gender")
    return df2


def test_perform_feature_engineering(df2):
    '''
    test perform_feature_engineering function
    '''
    test_size = 0.3
    random_state = 42
    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(df2, test_size, random_state)
    print("Running peform_feature_engineering function")

    try:
        assert 'Churn' not in X_train.columns
        logging.info("Churn column removed by perform_feature_engineering")
    except AssertionError as err:
        logging.error("Churn column not removed by perform_feature_engineering")
    try:
        assert 'CLIENTNUM' not in X_train.columns
        logging.info("CLIENTNUM column removed by perform_feature_engineering")
    except AssertionError as err:
        logging.error("CLIENTNUM column not removed by perform_feature_engineering")

    try:
        assert round(len(X_train)/len(df2), 2) == (1-test_size)
        logging.info("X_train has the correct number of rows")
    except AssertionError as err:
        logging.error("X_train does not have the correct number of rows")
    try:
        assert round(len(X_test)/len(df2), 2) == test_size
        logging.info("X_test has the correct number of rows")
    except AssertionError as err:
        logging.error("X_test does not have the correct number of rows")
    try:
        assert round(len(y_train)/len(df2), 2) == (1-test_size)
        logging.info("y_train has the correct number of rows")
    except AssertionError as err:
        logging.error("y_train does not have the correct number of rows")
    try:
        assert round(len(y_test)/len(df2), 2) == test_size
        logging.info("y_test has the correct number of rows")
    except AssertionError as err:
        logging.error("y_test does not have the correct number of rows")

def test_train_models(df2):
    '''
    test train_models function
    '''
    test_size = 0.3
    random_state = 42
    X_train, X_test, y_train, y_test = \
        cls.perform_feature_engineering(df2, test_size, random_state)
    image_dir_results = 'images_test/results/'
    cleanup_dir(image_dir_results)
    print("Running train_models function")
    cls.train_models(X_train, X_test, y_train, y_test,
                     image_dir_results=image_dir_results)

    try:
        assert os.path.isfile(image_dir_results + "classification_results.png")
        logging.info("classification_results.png is created")
    except AssertionError as err:
        logging.error("classification_results.png is not created")
    try:
        assert os.path.isfile(image_dir_results + "feature_importances_rfc.png")
        logging.info("feature_importances_rfc.png is created")
    except AssertionError as err:
        logging.error("feature_importances_rfc.png is not created")
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info("rfc_model.pkl is created")
    except AssertionError as err:
        logging.error("rfc_model.pkl is not created")
        