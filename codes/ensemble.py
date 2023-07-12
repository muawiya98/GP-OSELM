import warnings
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Redirect warnings to null device
with open(os.devnull, "w") as devnull:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("This is a sample warning.")  # Example warning that will be suppressed

        # Rest of your code

print("Print without warning")



from oselm import OSELMClassifier
from multiprocessing.pool import ThreadPool
import numpy as np 

class Classifier:
    def __init__(self, clf, max_number_of_classes:int=2):
        """
        Wrapping sklearn classifiers
        clf: sklearn classifiers like (KNN, LogRegression, DecisionTree, etc...)
        max_number_of_classes: integer, number of unique values in the predicted variable.
        """
        self.clf = clf
        # decision profile contains the prediction probability values.
        self.decision_profile = None
        self.max_number_of_classes = max_number_of_classes
        
    
    # fit the classifier
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
        self.apply_model_replacement = apply_model_replacement

    def fit(self, X_train, y_train, unselected_features=None):
        self.classifier_induction(self.classifiers, X_train, y_train, unselected_features=unselected_features)
        self.update_program(X_train, y_train)
    

    def classifier_induction(self, new_classifiers, X_train:np.array, y_train:np.array, unselected_features:list=None)->list:
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
        if self.apply_model_replacement:
          self.classifiers += trained_classifiers
        else:
          self.classifiers = trained_classifiers
        # return the trained classifiers (new classifiers after training)
        return trained_classifiers

    def model_replacement(self, criteria='best'):
        if criteria == 'best':
          pass
        elif criteria == 'time':
          self.classifiers = self.classifiers[3:]
            

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
        # accuracy_score, precision_score, recall_score, f1_score
        try:
          auc = roc_auc_score(y_test, y_pred)
        except:
          auc = 0.5
        self.scores[chunk_id] = {"accuracy": accuracy_score(y_test, y_pred),
                                 "precision": precision_score(y_test, y_pred),
                                 "recall": recall_score(y_test, y_pred), 
                                 "f1-score": f1_score(y_test, y_pred),
                                 "auc": auc}
        print(self.scores)
