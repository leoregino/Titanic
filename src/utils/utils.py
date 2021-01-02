import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

def get_size_family(df):

    """Defines family relations based on the features 'SibSp' (the # of siblings / spouses aboard the Titanic)
    and 'Parch' (the # of parents / children aboard the Titanic)

    Parameters
    ----------
    df : panda dataframe
    Returns
    -------
    New feature called FamilySize
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    return df

def modify_fare(df):
    """Introduce new intervals for the feature fare, such that it is modified from being continuous to be discrete

    Parameters
    ----------
    df : panda dataframe
    Returns
    -------
    Discretized version of the feature Fare
    """
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    return df

def get_title(name):
    """Search for individual title in a string by considering it to have a ASCII format from  A-Z

    Parameters
    ----------
    name : The name in which the title wants to be extracted (string)
    Returns
    -------
    Discretized version of the feature Fare
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)

    return ""

def get_titles(df, mod: bool = True):
    """Search for all titles inside a dataframe for the feature Name

    Parameters
    ----------
    df : panda dataframe
    mod : simplify the extend of titles available (boolean)
    Returns
    -------
    Discretized version of the feature Fare
    """
    df['Title'] = df['Name'].apply(get_title)
    if mod:
        # perform modifications
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df

def get_new_ages(df):
    """Search for all titles inside a dataframe for the feature Name

    Parameters
    ----------
    df : panda dataframe
    Returns
    -------
    Discretized version of the feature Fare
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
    df["Age"] = pd.cut(df["Age"], bins=[0, 16, 32, 48, 64, np.inf],
                       labels=['Childs', 'Young Adult', 'Adult', 'Senior', 'Oldy'], right=False)

    return df

def modify_titles(df):
    # join less representative cotegories
    df['Title'] = df['Title'].replace(['Lady', 'Countess',
                                       'Capt', 'Col', 'Don', 'Dr', 'Major',
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    return df

def get_deck(name):
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
    df['Deck'] = df['Cabin'].apply(get_deck)
    # Modifications
    df['Deck'] = df['Deck'].replace('T', 'None')
    return df

def embarked_bayes(df, i):
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
    emb = []
    for i, Port in df.iterrows():
        if pd.isnull(Port['Embarked']):
            emb.append(embarked_bayes(df, i))
        else:
            emb.append(Port['Embarked'])
    # Update column
    df['Embarked'] = emb
    return df

def transform_cat_to_num(df):
    # dictionary to transform gender to numerical
    sec_dic = {'male': 0, 'female': 1}
    # dictionary to transform ages to numerical
    sec_ages = {'Childs': 0, 'Young Adult': 1, 'Adult': 2, 'Senior': 3, 'Oldy': 4}
    # dictionary to transform embarked to numerical
    sec_emb = {'S': 0, 'C': 1, 'Q': 2}
    # dictionary to transform Deck to numerical
    sec_deck = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'None': 7}
    # dictionary to transform title to numerical
    sec_title = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4}
    # transform Sex to numerical
    df['Sex'] = df['Sex'].map(sec_dic).astype(int)
    # transform Age to numerical
    df['Age'] = df['Age'].map(sec_ages).astype(int)
    # transform Embarked to numerical
    df['Embarked'] = df['Embarked'].map(sec_emb).astype(int)
    # transform Deck to numerical
    df['Deck'] = df['Deck'].map(sec_deck).astype(int)
    # transform Title to numerical
    df['Title'] = df['Title'].map(sec_title).astype(int)

    return df

def drop_features(df, to_drop):

    return df.drop(to_drop, axis=1)


def pipeline_features(df, to_drop):
    # get new or modified features
    df = get_size_family(df)
    df = modify_fare(df)
    df = get_titles(df)
    df = get_new_ages(df)
    df = modify_titles(df)
    df = get_decks(df)
    df = get_embarked_bayes(df)
    # transformation of categorical to numerical
    df = transform_cat_to_num(df)
    # features to drop
    df = drop_features(df, to_drop)

    return df

