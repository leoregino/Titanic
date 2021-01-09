import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.utils import utils 

def transformation_features(df):
    """get new or modified features based on EDA

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Dataframe with all studied modifications
    """
    df = utils.get_size_family(df, mod = True)
    df = utils.modify_fare(df, 4)
    df = utils.get_titles(df, True)
    df = utils.get_all_ages(df, 5)
    df = utils.modify_titles(df)
    df = utils.get_decks(df)
    df = utils.get_embarked_bayes(df)
    # extra
    df = utils.get_if_cabin(df)
    df = utils.get_type_ticket(df)
    df = utils.get_count_name(df)

    return df

def pipeline_features(df, cat_ord_features, cat_hot_features, other_features):
    
    # Define ordinal categorical pipeline
    cat_ord_pipe = Pipeline([('encoder', OrdinalEncoder())])
    
    # Define categorical pipeline
    cat_hot_pipe = Pipeline([('encoder', OneHotEncoder(sparse=False))])
    
    # Define categorical pipeline
    other_pipe = Pipeline([('other_encoding', None)])

    # Fit column transformer to training data
    preprocessor = ColumnTransformer(transformers=[('cat_ord',  cat_ord_pipe, cat_ord_features),
                                                   ('cat',      cat_hot_pipe, cat_hot_features),
                                                   ('none',     other_pipe, other_features)])
    preprocessor.fit(df)

    # Prepare column names
    if len(cat_hot_features) != 0:
        cat_columns = preprocessor.named_transformers_['cat']['encoder'].get_feature_names(cat_hot_features)
        columns = np.hstack(( cat_ord_features, cat_columns, other_features )).ravel()
    else:
        columns = np.hstack(( cat_ord_features, other_features )).ravel()
    
    return pd.DataFrame(preprocessor.transform(df), columns=columns)