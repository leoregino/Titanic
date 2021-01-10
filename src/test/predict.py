import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlflow

def make_prediction(model, X_train_data, Y_train, X_test):
    """Add predictions from a specific model to the dataframe (test)

    Parameters
    ----------
    model                 : model considered for fitting training data (object)
    X_train_data, Y_train : array with training features and target labels
    frame                 : testing frame where to add predictions from trained model 
    
    Returns
    -------
    Original dataframe with new feature 'Survived', calculated by using a specific model
    """
    
    model.fit(X_train_data, Y_train)
    
    predictions = model.predict(X_test.drop(columns = ['PassengerId', 'Survived']))
    
    frame = pd.DataFrame(data={'PassengerId': X_test['PassengerId'].astype(int), 'Survived': predictions.astype(int)})
    
    return frame

def make_prediction_opt(prediction, X_test):
    """Add predictions from a specific model to the dataframe (test)

    Parameters
    ----------
    Prediction
    frame                 : testing frame where to add predictions from trained model 
    
    Returns
    -------
    Original dataframe with new feature 'Survived', calculated by using a specific model
    """
    
    frame = pd.DataFrame(data={'PassengerId': X_test['PassengerId'].astype(int), 'Survived': prediction.astype(int)})
    
    return frame

def save_experiment_mlflow(name_experiment, results, top_3: bool = True, end_run: bool = False):
        
    mlflow.set_experiment(name_experiment)
    
    if top_3:
        res = results.sort_values(by=['cv_acc'], ascending=False)[:3]
    
    for index, classifier in res.iterrows():
        
        with mlflow.start_run():
            
            # log parameters of interest
            mlflow.log_param("classifier", classifier['Model'])
            mlflow.log_param("best_params",  classifier['best_params'])
            # log metrics of interest
            mlflow.log_metric("CV_acc", classifier['cv_acc'])
            mlflow.log_metric("CV_acc_std", classifier['cv_acc_std'])

    if end_run == True:
        mlflow.end_run()

    return print("Training information saved")