'''
Library with all the functions needed for churn prediction

Author: Dallas Lea
August 2023
'''

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import joblib

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

IMAGE_DIR_EDA = './images_final/eda/'
IMAGE_DIR_RESULTS = './images_final/results/'
MODEL_DIR = './models_final/'
CATEGORY_LST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df, image_dir_eda):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            image_dir_eda: images folder

    output:
            None
    '''

    print(f"shape: {df.shape}")
    print("\nNull counts:")
    print(df.isnull().sum())
    print("\nSome statistics:")
    print(df.describe())

    plot_configs = [
        {'name': "Churn histogram",
         'plot_fn': lambda: df['Churn'].hist(),
         'path': os.path.join(image_dir_eda, 'churn_hist.png')},
        {'name': "Customer Age histogram",
         'plot_fn': lambda: df['Customer_Age'].hist(),
         'path': os.path.join(image_dir_eda, 'age_hist.png')},
        {'name': "Marital_Status barchart",
         'plot_fn': lambda: df.Marital_Status.value_counts('normalize').plot(kind='bar'),
         'path': os.path.join(image_dir_eda, 'marital_bar.png')},
        {'name': "Total_Trans_Ct density plot",
         'plot_fn': lambda: sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True),
         'path': os.path.join(image_dir_eda, 'ttc_dens.png')},
        {'name': "Correlation plot",
         'plot_fn': lambda: sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2),
         'path': os.path.join(image_dir_eda, 'corrplot.png'),
         'size': (20, 10)},
    ]
    for config in plot_configs:
        if 'size' in config:
            plt.figure(figsize=config['size'])
        else:
            plt.figure(figsize=(8, 5))
        plot = config['plot_fn']()
        plt.tight_layout()
        fig = plot.get_figure()

        # Create directories if they don't exist
        _create_dirs(config['path'])

        fig.savefig(config['path'])
        plt.close(fig)
        print(f"Saved {config['name']} to {config['path']}")

def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    df = pd.get_dummies(df, columns=category_lst)

    return df

def perform_feature_engineering(df, test_size, random_state):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count',
        'Months_on_book', 'Total_Relationship_Count',
        'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio', 'Gender_F', 'Gender_M',
        'Education_Level_College', 'Education_Level_Doctorate',
        'Education_Level_Graduate', 'Education_Level_High School',
        'Education_Level_Post-Graduate', 'Education_Level_Uneducated',
        'Education_Level_Unknown', 'Marital_Status_Divorced',
        'Marital_Status_Married', 'Marital_Status_Single',
        'Marital_Status_Unknown', 'Income_Category_$120K +',
        'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
        'Income_Category_$80K - $120K', 'Income_Category_Less than $40K',
        'Income_Category_Unknown', 'Card_Category_Blue',
        'Card_Category_Gold', 'Card_Category_Platinum',
        'Card_Category_Silver']
    X[keep_cols] = df[keep_cols]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # scores
    print('****Random Forest Results****')
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))

    print('****Logistic Regression Results****')
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))

    # Create & store plot
    plt.figure(figsize=(5, 9))

    plt.text(0.01, 1.00, 'Random Forest Train Results', {'fontsize': 12},
        fontproperties='monospace')
    plt.text(0.01, 0.775, str(classification_report(y_train, y_train_preds_rf)),
        {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.725, 'Random Forest Test Results', {'fontsize': 12},
        fontproperties='monospace')
    plt.text(0.01, 0.5, str(classification_report(y_test, y_test_preds_rf)),
        {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.45, 'Logistic Regression Train Results', {'fontsize': 12},
        fontproperties='monospace')
    plt.text(0.01, 0.225, str(classification_report(y_train, y_train_preds_lr)),
        {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.175, 'Logistic Regression Test Results', {'fontsize': 12},
        fontproperties='monospace')
    plt.text(0.01, -0.05, str(classification_report(y_test, y_test_preds_lr)),
        {'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X value
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.tight_layout()
    plt.savefig(output_pth, bbox_inches='tight')

def train_models(X_train, X_test, y_train, y_test,
                 image_dir_results=IMAGE_DIR_RESULTS):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
              param_grid_rfc: parameters to pass into the GridSearchCV's
                  `param_grid` argument that uses RandomForestClassifier.
              image_dir_results: Directory that stores the result images
              model_dir: Directory that stores the model
    output:
              None
    '''

    # This cell may take up to 15-20 minutes to run
    # train test split

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)
    # Check the convergence status
    if lrc.n_iter_ < lrc.max_iter:
        print("LRC model converged successfully.")
    else:
        print("LRC model did not converge.")


    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Create & store ROC curve and summary plot
    _create_dirs(image_dir_results)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.show

    # Create directories if they don't exist
    _create_dirs(image_dir_results)

    path = os.path.join(image_dir_results, 'roc_curve.png')
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved ROC curve to {path}")
    plt.close()

    # Store classification report as image
    output_pth = os.path.join(image_dir_results, 'classification_results.png')
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth=output_pth)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Create & store feature importances
    fir_path = os.path.join(image_dir_results, 'feature_importances_rfc.png')
    feature_importance_plot(cv_rfc.best_estimator_, X_train, fir_path)


def _create_dirs(filepath):
    # Create directories if they don't exist
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    DATA_PATH = r"./data/bank_data.csv"

    df = import_data(DATA_PATH)
    EDA = perform_eda(df, IMAGE_DIR_EDA)
    df = encoder_helper(df, CATEGORY_LST)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, test_size = 0.3, random_state = 42)
    train_models(X_train, X_test, y_train, y_test)
