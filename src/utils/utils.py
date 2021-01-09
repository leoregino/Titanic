import os, glob, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import re

def load_data(path):
    """Load training and testing datasets based on their path

    Parameters
    ----------
    path : relative path to location of data, should be always the same (string)
    
    Returns
    -------
    Training and testing Dataframes
    """
    train = pd.read_csv(os.path.join(path,'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))
    
    return train, test
    
def modify_fare(df, n: int = 4):
    """Introduce n new intervals (based on quantiles) for the feature fare, such that it is modified from
    being continuous to being discrete

    Parameters
    ----------
    df : panda dataframe
    n: number of new intervals (int)
    
    Returns
    -------
    Original dataframe with discretized version of the feature 'Fare', categories
    """
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare'] = pd.qcut(df['Fare'], n, labels = list(string.ascii_uppercase)[:n])
    
    return df

def get_size_family(df, mod: bool = False):
    """Defines family relations based on the features 'SibSp' (the # of siblings / spouses aboard the Titanic)
    and 'Parch' (the # of parents / children aboard the Titanic)

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with a new feature called 'FamilySize'
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    if mod:
        
        bins_ = [0,1,2,12]
        df['FamilySize'] = pd.cut(df["FamilySize"],  bins = bins_, labels = list(string.ascii_uppercase)[:len(bins_)-1])
        
    return df

def get_title(name):
    """Search for individual title in a string by considering it to have a ASCII format from  A-Z

    Parameters
    ----------
    name : The name from which a title wants to be extracted (string)
    
    Returns
    -------
    String associated to a found title
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)

    return ""

def get_titles(df, mod: bool = True):
    """Search for all titles inside a dataframe, given the feature 'Name'

    Parameters
    ----------
    df : panda dataframe
    mod : simplify the extend of titles available (boolean)
    
    Returns
    -------
    Original dataframe with a new feature called 'Title'
    """
    df['Title'] = df['Name'].apply(get_title)
    if mod:
        # perform modifications
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df

def get_all_ages(df, n: int = 5):
    """Fills in empty Ages based on the Title of a person, and then introduces n intervals for the feature 'Ages', 
    such that it is modified from being continuous to be discrete

    Parameters
    ----------
    df : panda dataframe
    n: number of new intervals (int)
    
    Returns
    -------
    Discretized version of the feature 'Age', categories
    """
    emb = []

    for i, row in df.iterrows():
        if pd.isnull(row['Age']):
            title = row['Title']
            age_avg = df['Age'][df['Title'] == title].mean()
            age_std = df['Age'][df['Title'] == title].std()
            emb.append(np.random.randint(age_avg - age_std, age_avg + age_std, size=1)[0])
        else:
            emb.append(row['Age'])

    # Update column
    df['Age'] = emb
    # Create new column
    df["Age"] = pd.cut(df["Age"], n, labels = list(string.ascii_uppercase)[:n])

    return df

def modify_titles(df):
    """Concatenates titles found to be similar or considered to be simplified in one category

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Simplified categories in the features 'Title'
    """
    # join less representative cotegories
    df['Title'] = df['Title'].replace(['Lady', 'Countess',
                                       'Capt', 'Col', 'Don', 'Dr', 'Major',
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    return df

def get_deck(name):
    """Search for individual Capital letter inside a string associated to the cabin of a person, from  A-Z

    Parameters
    ----------
    name : The name from which a deck wants to be extracted (string)
    
    Returns
    -------
    Letter associated with the deck from that a person has
    """    
    if pd.isnull(name):
        return 'None'
    else:
        title_search = re.findall(r"^\w", name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search[0]
        else:
            return 'None'

def get_decks(df):
    """Search for the information of all decks inside a dataframe, given the feature 'Cabin'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with a new feature called 'Deck'
    """
    
    df['Deck'] = df['Cabin'].apply(get_deck)
    # Modifications
    df['Deck'] = df['Deck'].replace('T', 'None')
    return df

def embarked_bayes(df, i):
    """Using Bayes Theorem, and based on 'Pclass', determine the probability of 'Embarked' for a person 
    given the possibilities S, C or Q.

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    String associated to the most likely port from where a passenger Embarked, given its Pclass
    """
    
    pclass_ = df['Pclass'].iloc[i]
    # P(s|1) = P(s)*P(1|S)/[ P(s)*P(1|s) + P(s)*P(1|s) + P(s)*P(1|s)] # probability that given the class 1, the person came from port S
    P_S, P_C, P_Q = df['Embarked'].value_counts()['S'], df['Embarked'].value_counts()['C'], \
                    df['Embarked'].value_counts()['Q']
    P_class_S = df['Embarked'][df['Pclass'] == pclass_].value_counts()['S']
    P_class_C = df['Embarked'][df['Pclass'] == pclass_].value_counts()['C']
    P_class_Q = df['Embarked'][df['Pclass'] == pclass_].value_counts()['Q']
    res = []
    P_S_class = (P_S * P_class_S) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_S_class)
    P_C_class = (P_C * P_class_C) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_C_class)
    P_Q_class = (P_Q * P_class_Q) / ((P_S * P_class_S) + (P_C * P_class_C) + (P_Q * P_class_Q))
    res.append(P_C_class)

    if sorted(res, reverse=True)[0] == P_S_class:
        return 'S'
    elif sorted(res, reverse=True)[0] == P_C_class:
        return 'C'
    elif sorted(res, reverse=True)[0] == P_Q_class:
        return 'Q'

def get_embarked_bayes(df):
    """Search for the Embarked information of passengers missing this data, based on its 'Pclass'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Original dataframe with all missing values from the feature 'Embarked'
    """
    emb = []
    for i, Port in df.iterrows():
        if pd.isnull(Port['Embarked']):
            emb.append(embarked_bayes(df, i))
        else:
            emb.append(Port['Embarked'])
    # Update column
    df['Embarked'] = emb
    return df

def get_if_cabin(df):
    """Indicate if a person has a 'Cabin'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    String with a Yes or No
    """
    # Feature that tells whether a passenger had a cabin on the Titanic
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 'No' if type(x) == float else 'Yes')
    
    return df

def get_type_ticket(df):
    """Indicate if a person has a 'Ticket'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Categorical unique code
    """
    # Feature that tells whether a passenger had a cabin on the Titanic
    df['Type_Ticket'] = df['Ticket'].apply(lambda x: x[0:3])
    df['Type_Ticket'] = df['Type_Ticket'].astype('category').cat.codes # ordinal encoding
    df['Type_Ticket'] = df['Type_Ticket'].astype(int)
    return df

def get_count_name(df):
    """Indicate if a person has a 'Name'

    Parameters
    ----------
    df : panda dataframe
    
    Returns
    -------
    Categorical unique code
    """
    # Feature that tells whether a passenger had a cabin on the Titanic
    df['Words_Count'] = df['Name'].apply(lambda x: len(x.split())).astype(int)
    
    return df

def drop_features(df, to_drop):
    """Drop unwanted features 

    Parameters
    ----------
    df      : panda dataframe
    to_drop : array with name of features to be dropped
    
    Returns
    -------
    Original dataframe with all original features but those in to_drop
    """
    return df.drop(to_drop, axis=1)