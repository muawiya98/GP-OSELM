from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import ListedColormap
from multiprocessing.pool import ThreadPool
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from contextlib import suppress
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from random import shuffle
import seaborn as sns
from time import time
import pandas as pd
import numpy as np
import warnings
import scipy.io
import pickle
import sys
import gc
import os
import re


def prepare_data(csv_filename, target_column_name='class'):
    # read csv file
    df = pd.read_csv(csv_filename)
    df = df.iloc[:80000, :]
    column_names = df.columns.tolist()
    if target_column_name not in column_names:
        target_column_name = column_names[-1]
    # get unique value in target column
    unique_vlaues = sorted(df[target_column_name].unique().tolist())
    df[target_column_name] = df[target_column_name].apply(lambda x: 0 if x == unique_vlaues[0] else 1)
    df[target_column_name] = df[target_column_name].astype('int')
    # rename the column of the dataframe
    num_of_columns = len(column_names)
    df.columns = list(range(num_of_columns))
    return df
  
  
  
def train_and_test(model, X_train, y_train, X_test, y_test, unselected_features=None):
    model.fit(X_train, y_train, unselected_features)
    y_pred = model.predict(X_test)
    model.fit(X_test, y_test, unselected_features)
    return model, y_pred
  
def feature_evolving(evolving_matrix):
    """
    evolving_matrix : list of random list
    """
    random_index = np.random.randint(0, len(evolving_matrix), 1)[0]
    return evolving_matrix[random_index]
  
def save_pickle(obj, file_name):
  with open(file_name, 'wb') as f:
    pickle.dump(obj, f)
def load_pickle(file_name):
  with open(file_name, 'rb') as f:
    d = pickle.load(f)
  return d

def save_object(obj, filename,path):
    """
    _ INPUT (obj) THE OBJECT WE NEED SAVW IT (filename) THE NAME OF OBJECT
    """
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()
def load_object(filename,path):
    """
    _ INPUT THE NAME OF OBJECT WE NEED LOAD IT
    """
    filename = os.path.join(path,filename)
    with open(filename+".pkl", 'rb') as outp:
        loaded_object = pickle.load(outp)
    outp.close()
    return loaded_object
  
  
def generate_new_samples(buffer, y_values, n=500, y_col='label'):
    if not y_col in buffer.columns.tolist():
      y_col = buffer.columns.tolist()[-1]
    if y_values.sum() == 0:
       return buffer[buffer[y_col] == 1].sample(n, random_state=41)[:, :-1].values, np.array([1] * n)
    else:
      return buffer[buffer[y_col] == 0].sample(n,random_state=41)[:, :-1].values, np.array([0] * n)
    
    
    
    
