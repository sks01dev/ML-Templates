"""
Multilabel Classification:
- each example may belong to multiple classes
- eg. detecting multiple objects in one image example
"""

# Precision @ k (P@k)
def pk(y_true, y_pred, k):
    """ 
    This function calculates precision at k  
    for a single sample 
    :param y_true: list of values, actual classes 
    :param y_pred: list of values, predicted classes 
    :return: precision at a given value k 
    """ 
    # if k is 0, return 0. we should never have this 
    # as k is always >= 1 
    if k == 0: 
        return 0 
    # we are interested only in top-k predictions 
    y_pred = y_pred[:k] 
    # convert predictions to set 
    pred_set = set(y_pred) 
    # convert actual values to set 
    true_set = set(y_true) 
    # find common values 
    common_values = pred_set.intersection(true_set) 
    # return length of common values over k, y_pred[:k] in case k=5 but len(y_pred) = 2? -> /2 
    return len(common_values) / len(y_pred[:k]) 



# Average Precision @ k (AP@k) : average over P@i for i in 1-k

def apk(y_true, y_pred, k): 
    """ 
    This function calculates average precision at k  
    for a single sample 
    :param y_true: list of values, actual classes 
    :param y_pred: list of values, predicted classes 
    :return: average precision at a given value k 
    """ 
    # initialize p@k list of values 
    pk_values = [] 
    # loop over all k. from 1 to k + 1 
    for i in range(1, k + 1): 
        # calculate p@i and append to list 
        pk_values.append(pk(y_true, y_pred, i)) 
        # if we have no values in the list, return 0 
        if len(pk_values) == 0: 
            return 0 
    # else, we return the sum of list over length of list 
    return sum(pk_values) / len(pk_values) 


# Mean Average Precision @ k (MAP@k): mean of AP@k over all the examples

def mapk(y_true, y_pred, k): 
    """ 
    This function calculates mean avg precision at k  
    for a single sample 
    :param y_true: list of values, actual classes 
    :param y_pred: list of values, predicted classes 
    :return: mean avg precision at a given value k 
    """ 
    # initialize empty list for apk values 
    apk_values = [] 
    # loop over all samples 
    for i in range(len(y_true)): 
        # store apk values for every sample 
        apk_values.append(apk(y_true[i], y_pred[i], k=k)) 
    # return mean of apk values list 
    return sum(apk_values) / len(apk_values)



"""
Multi-label log loss =
average(
    binary log loss per label
)
Also known as mean column wise log loss.
"""

def multi_log_loss(y_true, y_pred):
    """
    Calculates the mean column wise log loss for a single example.
    Averages logloss over all columns/labels.
    """
    # Accumulate losses for every column
    losses = [] 
    for col in range(y_true.shape[1]):
        # take one col/labels from y_true and same from y_pred, then feed into logloss
        losses.append(sklearn.metrics.log_loss(y_true[:, col], y_pred[:, col]))

    final_loss = sum(losses) / len(losses)
    return final_loss

"""
# 2 samples, 2 labels
y_true = np.array([
    [1, 0],
    [0, 1]
])

y_pred = np.array([
    [0.9, 0.2],
    [0.1, 0.8]
])

print(multi_log_loss(y_true, y_pred))

logloss b/w 1st col/label in y_true vs 1st col/label in y_pred
i.e [1, 0] and [0.9, 0.1]
"""
