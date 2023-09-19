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
import datetime
import pickle
import sys
import gc
import os
import re 

code_path = '/content/drive/My Drive/Colab Notebooks/Muawiya/Genetic Programming Combiner with DFS/codes/Shared Codes'
sys.path.insert(0,code_path)
from oselm import OSELMClassifier,set_use_know
class Classifier:
    def __init__(self, clf, max_number_of_classes:int=2):
        """
        Wrapping sklearn classifiers
        clf: sklearn classifiers like (KNN, LogRegression, DecisionTree, etc...)
        max_number_of_classes: integer, number of unique values in the predicted variable.
        """
        self.clf = clf
        self.decision_profile = None
        self.max_number_of_classes = max_number_of_classes


    def fit(self, X_train, y_train, unselected_features=None):
        """
        Call the training function
        X_train: 2d array with shape num_of_samples x num_of_feautres.
        y_train: 1d array with shape (num_of_samples, ) contains the ground truth values.
        """
        if type(self.clf) == OSELMClassifier:
            self.clf.fit(X_train, y_train, unselected_features)
        else:
            self.clf.fit(X_train, y_train)

    def predict_proba(self, X):
        """
        predict the probability of belonging this `sample` to each class
        """
        # sometimes number of unique values in the predicted variable differ from one chunk to another,
        # so that we need to pad the results of probablity prediction to new size equal to `max_number_of_classes`

        pred = self.clf.predict_proba(X)
        return pred

    def build_decision_profile(self, sample):
        """
        add the predict_probability result to the `decision_profile` list
        sample: one example form the dataset
        """
        self.decision_profile = self.predict_proba(sample.reshape((1, -1)))[0].tolist()


class Ensemble:
    def __init__(self, classifiers, program, apply_model_replacement):

        """
        classfiers : list of Classifier objects
        program: result of genetic programming (SymbolicRegressor)
        """
        self.classifiers = classifiers
        self.program = program
        self.program_history = []
        self.fitted = False
        self.scores = {}
        self.maximum_number_of_classifer = 4
        self.apply_model_replacement = apply_model_replacement

    def get_scores(self):
      return self.scores
    
    def set_scores(self,scores):
      self.scores = scores
    
    def fit(self, X_train, y_train, unselected_features=None):
        self.classifier_induction(self.classifiers, X_train, y_train, unselected_features=unselected_features,only_fit=True)
        self.update_program(X_train, y_train)


    def classifier_induction(self, new_classifiers, X_train:np.array, y_train:np.array, unselected_features:list=None,only_fit=False)->list:
        """
        new_classifiers: list of new classifiers to insert them into ensemble classifiers.
        X_train: training dataset .
        y_train: ground truth values.
        unselected_features: indices of unselected features at each chunk
        ----------------------------------------------------------------
        return new_classifiers after training.
        """
        # use classifier_induction_util for multiprocessing
        def classifier_induction_util(classifier):
            clf = Classifier(classifier, 2)
            clf.fit(X_train.copy(), y_train.copy(), unselected_features)
            return clf
        # train each new classifier in parallel
        trained_classifiers = ThreadPool(len(new_classifiers)).map(classifier_induction_util, new_classifiers)
        # add the trained classifiers to the ensemble classifiers.
        if self.apply_model_replacement and not only_fit:
          self.classifiers += trained_classifiers
        else:
          self.classifiers = trained_classifiers
        # return the trained classifiers (new classifiers after training)
        return trained_classifiers

    def model_replacement(self, criteria='best'):
        if criteria == 'best':
          pass
        elif criteria == 'time':
          self.classifiers = self.classifiers[-self.maximum_number_of_classifer:]


    def global_support_degree(self, sample):
        for i,clf in enumerate(self.classifiers):
            if not isinstance(clf,Classifier):
              clf = Classifier(clf,2)
              self.classifiers[i] = clf
            clf.build_decision_profile(sample)
        profile = np.array([self.classifiers[i].decision_profile for i in range(len(self.classifiers))])
        return np.argmax(profile.sum(axis=0))

    def update_program(self, X, y):
        # change the fit flag to True.
        self.fitted = True
        profiles = np.array([self.classifiers[i].predict_proba(X) for i in range(len(self.classifiers))])
        self.program.fit(profiles, y)
        self.program_history.append(self.program)


    def predict(self, X_test):
        X_test = np.squeeze(X_test) if len(list(X_test.shape))>2 else X_test
        profiles = np.array([self.classifiers[i].predict_proba(X_test) for i in range(len(self.classifiers))])
        return self.program.predict(profiles)

    def evaluate(self, X_test, y_test, chunk_id=1):
        y_pred = self.predict(X_test)
        try:
          auc = roc_auc_score(y_test, y_pred)
        except:
          auc = 0.5
        self.scores[chunk_id] = {"accuracy": accuracy_score(y_test, y_pred),
                                 "precision": precision_score(y_test, y_pred),
                                 "recall": recall_score(y_test, y_pred),
                                 "f1-score": f1_score(y_test, y_pred),
                                 "auc": auc}
        print(chunk_id,self.scores[chunk_id])
