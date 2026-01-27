# Multiclass classification metrics

import numpy as np 
import sklearn

from collections import Counter 

## Accumulate the precision for each class and then average them. 
def macro_precision(y_true, y_pred): 
    """ 
    Function to calculate macro averaged precision 
    :param y_true: list of true values 
    :param y_proba: list of predicted values 
    :return: macro precision score 
    """ 
     
    # find the number of classes by taking 
    # length of unique values in true list 
    num_classes = len(np.unique(y_true)) 
     
    # initialize precision to 0 
    precision = 0 
     
    # loop over all classes 
    for class_ in range(num_classes): 
         
        # all classes except current are considered negative 
        temp_true = [1 if p == class_ else 0 for p in y_true] 
        temp_pred = [1 if p == class_ else 0 for p in y_pred] 
         
        # calculate true positive for current class 
        tp = true_positive(temp_true, temp_pred) 
         
        # calculate false positive for current class 
        fp = false_positive(temp_true, temp_pred) 
         
        # calculate precision for current class 
        temp_precision = tp / (tp + fp) 
         
        # keep adding precision for all classes 
        precision += temp_precision 
 
    # calculate and return average precision over all classes 
    precision /= num_classes 
    return precision 


## Get the true and false positives for each class and accumulate them. Then calculate
## micro precision.
def micro_precision(y_true, y_pred): 
    """ 
    Function to calculate micro averaged precision 
    :param y_true: list of true values 
    :param y_proba: list of predicted values 
    :return: micro precision score 
    """ 
    # find the number of classes by taking 
    # length of unique values in true list 
    num_classes = len(np.unique(y_true)) 
    
    # initialize tp and fp to 0 
    tp = 0 
    fp = 0 
    
    # loop over all classes 
    for class_ in range(num_classes): 
        # all classes except current are considered negative 
        temp_true = [1 if p == class_ else 0 for p in y_true] 
        temp_pred = [1 if p == class_ else 0 for p in y_pred] 
        
        # calculate true positive for current class 
        # and update overall tp 
        tp += true_positive(temp_true, temp_pred) 
        
        # calculate false positive for current class 
        # and update overall tp 
        fp += false_positive(temp_true, temp_pred) 
    
    # calculate and return overall precision 
    precision = tp / (tp + fp) 
    return precision 

 
# Weighted precision i.e wrt the number of samples for each class
def weighted_precision(y_true, y_pred): 
    """ 
    Function to calculate weighted averaged precision 
    :param y_true: list of true values 
    :param y_proba: list of predicted values 
    :return: weighted precision score 
    """ 
     
    # find the number of classes by taking 
    # length of unique values in true list 
    num_classes = len(np.unique(y_true)) 
     
    # create class:sample_count dictionary 
    class_counts = Counter(y_true) 
     
    # initialize precision to 0 
    precision = 0 
     
    # loop over all classes 
    for class_ in range(num_classes): 
        # all classes except current are considered negative 
        temp_true = [1 if p == class_ else 0 for p in y_true] 
        temp_pred = [1 if p == class_ else 0 for p in y_pred] 
         
        # calculate tp and fp for class 
        tp = true_positive(temp_true, temp_pred) 
        fp = false_positive(temp_true, temp_pred) 
         
        # calculate precision of class 
        temp_precision = tp / (tp + fp) 
         
        # multiply precision with count of samples in class 
        weighted_precision = class_counts[class_] * temp_precision 
         
        # add to overall precision 
        precision += weighted_precision 
    # calculate overall precision by dividing by 
    # total number of samples 
    overall_precision = precision / len(y_true)
    return overall_precision

## Sklearn implementation

## sklearn.metrics.precision_score(y_true, y_pred, average='macro')

## You can change the average parameter with 'macro', 'micro', 'weighted'

"""
For recall, change the following:

- get the true positive and false negative counts
- use the formula to calculate recall, i.e tp / (tp + fn)
"""


# Similar implementation for weighted F1 score

def weighted_f1(y_true, y_pred): 
    """ 
    Function to calculate weighted f1 score 
    :param y_true: list of true values 
    :param y_proba: list of predicted values 
    :return: weighted f1 score 
    """ 
     
    # find the number of classes by taking 
    # length of unique values in true list 
    num_classes = len(np.unique(y_true)) 
     
    # create class:sample count dictionary 
    # it looks something like this: 
    # {0: 20, 1:15, 2:21} 
    class_counts = Counter(y_true) 
     
    # initialize f1 to 0 
    f1 = 0 
     
    # loop over all classes 
    for class_ in range(num_classes): 
        # all classes except current are considered negative 
        temp_true = [1 if p == class_ else 0 for p in y_true] 
        temp_pred = [1 if p == class_ else 0 for p in y_pred] 
         
        # calculate precision and recall for class 
        p = precision(temp_true, temp_pred) 
        r = recall(temp_true, temp_pred) 
         
        # calculate f1 of class 
        if p + r != 0: 
            temp_f1 = 2 * p * r / (p + r) 
        else: 
            temp_f1 = 0 
         
        # multiply f1 with count of samples in class 
        weighted_f1 = class_counts[class_] * temp_f1 
         
        # add to f1 precision 
        f1 += weighted_f1 
     
    # calculate overall F1 by dividing by 
    # total number of samples 
    overall_f1 = f1 / len(y_true) 
    return overall_f1 


##########################################
# This format of conversion from binary to multi-class is also called `one vs all`.
# FP -> Type 1 error, FN -> Type 2 error
