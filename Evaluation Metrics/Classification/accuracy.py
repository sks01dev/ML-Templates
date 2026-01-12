def accuracy(y_true, y_pred):
    """ 
    Function to calculate accuracy 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: accuracy score 
    """ 

    correct_counter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1

    """
    Alternative:
    use scikit-learn library
    """
    from sklearn import metrics
    acc = metrics.accuracy_score(y_true, y_pred)
    
    return correct_counter / len(y_true)
