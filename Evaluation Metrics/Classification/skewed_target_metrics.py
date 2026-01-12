# If the target column is skewed, then accuracy ain't reliable
# Then we can use Precision, Recall, F1

def true_positive(y_true, y_pred): 
    """ 
    Function to calculate True Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of true positives 
    """ 
    # initialize 
    tp = 0 
    for yt, yp in zip(y_true, y_pred): 
        if yt == 1 and yp == 1: 
            tp += 1 
    return tp 



def true_negative(y_true, y_pred): 
    """ 
    Function to calculate True Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of true negatives 
    """ 
    # initialize 
    tn = 0 
    for yt, yp in zip(y_true, y_pred): 
        if yt == 0 and yp == 0: 
            tn += 1 
    return tn 



def false_positive(y_true, y_pred): 
    """ 
    Function to calculate False Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false positives 
    """ 
    # initialize 
    fp = 0 
    for yt, yp in zip(y_true, y_pred): 
        if yt == 0 and yp == 1: 
            fp += 1 
    return fp 



def false_negative(y_true, y_pred): 
    """ 
    Function to calculate False Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false negatives 
    """ 
    # initialize 
    fn = 0 
    for yt, yp in zip(y_true, y_pred): 
        if yt == 1 and yp == 0: 
            fn += 1 
    return fn 


## Now calculating accuracy using the above functions
## Note: you can still use metrics.accuracy_score

def accuracy_v2(y_true, y_pred): 
    """ 
    Function to calculate accuracy using tp/tn/fp/fn 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: accuracy score 
    """ 
    tp = true_positive(y_true, y_pred) 
    fp = false_positive(y_true, y_pred) 
    fn = false_negative(y_true, y_pred) 
    tn = true_negative(y_true, y_pred) 
    accuracy_score = (tp + tn) / (tp + tn + fp + fn) 
    return accuracy_score 

def precision(y_true, y_pred): 
    """ 
    Function to calculate precision 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: precision score 
    """ 
    tp = true_positive(y_true, y_pred) 
    fp = false_positive(y_true, y_pred) 
    precision = tp / (tp + fp) 
    return precision 


def recall(y_true, y_pred): 
    """ 
    Function to calculate recall 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: recall score 
    """ 
    tp = true_positive(y_true, y_pred) 
    fn = false_negative(y_true, y_pred) 
    recall = tp / (tp + fn) 
    return recall 


# Harmonic mean of precison and recall -> F1
# Can be implemented from metrics.f1_score
def f1(y_true, y_pred): 
    """ 
    Function to calculate f1 score 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: f1 score 
    """ 
    p = precision(y_true, y_pred) 
    r = recall(y_true, y_pred) 
    score = 2 * p * r / (p + r) 
    return score 

def tpr(y_true, y_pred): 
    """ 
    Function to calculate tpr 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: tpr/recall 
    Note: TPR(True Positive Rate) is same as recall = SENSITIVITY
    """ 
    return recall(y_true, y_pred)

def fpr(y_true, y_pred): 
    """ 
    Function to calculate false positive rate 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: fpr 
    """ 
    fp = false_positive(y_true, y_pred) 
    tn = true_negative(y_true, y_pred) 
    return fp / (tn + fp) 

## For ROC-AUC you can use, metrics.roc_auc_score from Sklearn
"""
Select the best threshold from the leftmost top point in the ROC curve 
X-axis: FPR, Y-axis: TPR 
"""
