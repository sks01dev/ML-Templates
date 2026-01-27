import numpy as np 
 
 
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


