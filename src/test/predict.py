import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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