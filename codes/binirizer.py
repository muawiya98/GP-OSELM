
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




import numpy as np

class CustomLabelBinirizer:
    def __init__(self, negative_label:int=0, positive_label:int=1):
        """
        """
        self.negative_label = negative_label
        self.positive_label = positive_label
        self.transformer = None
        self.is_fit = False
        
    def fit(self, y):
        """
        """
        if self.is_fit:return
        if type(y) is list:
            y = np.array(y)
        ##############
        y_unique = np.unique(y)
        # print("what the bug {}".format(len(y_unique)))
        y_unique.sort()
        _shape = len(y_unique)
        ##############
        self.transformer = np.zeros((_shape, _shape))
        for unique_value in y_unique:
            self.transformer[unique_value, unique_value] = 1
        self.is_fit = True
        return 
    
    def transform(self, y):
        """
        """
        if not self.is_fit:
            self.fit(y)
        y_bin = []
        for value in y:
            y_bin.append(self.transformer[value, :].tolist())
        return np.array(y_bin)
            
        
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y_bin):
        return np.argmax(y_bin, axis=1)