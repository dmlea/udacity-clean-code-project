# udacity-clean-code-project
Final project for Udacity Clean Code Principles program

## Project Description

This project is a Python package for training and testing two models (Random Forest Classifier, Logistic Regression) to identify credit card customers likely to churn. Its machine learning code follows coding (PEP8) and engineering best practices for implementing software (modularity, testing, and documentation).

## Files and data description

- `data/`
  - `bank_data.csv`: The dataset for this project.

- `images_final/`
  - `eda/`
    - `age_hist.png`: Ages histogram
    - `churn_hist.png`: Churn histogram (0 for no churn, 1 for churn)
    - `corrplot.png`: Heat map of correlations between all numerical variables
    - `marital_bar.png`: Marital status distribution bar plot
    - `ttc_dens.png`: Total transaction distribution plot
  - `results/`
    - `classification_results.png`: Precision/Recall/F1 results of classifications
    - `feature_importances_rfc.png`: Random Forest model feature importances bar plot
    - `roc_curve.png`: ROC (Receiver Operating Characterstic) curves for both models
   
- `images_test/*/*`: same as `images_final/*/*` but for the unit tests      

- `models/`:
  - `rfc_model.pkl`: File containg random forest classifier model
  - `logistic_model.pkl`: File containing logistic regression model

- `churn_library.py`: Refactored script to execute same tasks as churn_notebook.ipynb
- `churn_script_logging_and_testing.py`: Unit testing file. Uses `pytest`

- `README.md`: This documentation file.
- `requirements_py3.8.txt`: Python requirements file for Python 3.8.
- `pytest.ini`: Configuration file for pytest.

- `logs/`:
  - `churn_library_test.log`: log file showing successful results from try/except assertions

## Running Files

The project was tested with Python 3.8. Run the following command to install all required modules:

```
pip install -r requirements_py3.8.txt
```

For unit tests with `pytest`, run this command:

```
pytest churn_script_logging_and_testing.py
```

A log of the pytest run will be stored in the `./logs/` directory.


To run all functions in the library, run this command:

```
python churn_library.py
```

## Code Quality Metrics

### Follows PEP 8 Guidelines

- Running `pylint ./churn_library.py` resulted in a score of 7.22
- Running `pylint ./churn_script_logging_and_tests.py` resulted in a score of 7.14


### Runs Successfully

Proven by contents of images_test/* and images_final/* directories.
