
import numpy as np 

def mean_absolute_error(y_true, y_pred): 
    """ 
    This function calculates mae 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: mean absolute error 
    """ 
    # initialize error at 0 
    error = 0 
    # loop over all samples in the true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate absolute error  
        # and add to error 
        error += np.abs(yt - yp) 
    # return mean error 
    return error / len(y_true)

"""
Alternative:

def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
"""

######################################

def mean_squared_error(y_true, y_pred): 
    """ 
    This function calculates mse 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: mean squared error 
    """ 
    # initialize error at 0 
    error = 0 
    # loop over all samples in the true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate squared error  
        # and add to error 
        error += (yt - yp) ** 2 
    # return mean error 
    return error / len(y_true) 

#####################################


def mean_squared_log_error(y_true, y_pred): 
    """ 
    This function calculates msle 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: mean squared logarithmic error 
    """ 
    # initialize error at 0 
    error = 0 
    # loop over all samples in true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate squared log error  
        # and add to error 
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2 
    # return mean error 
    return error / len(y_true) 

# Root mean squared logarithmic error is just a square root of this. It is also known as RMSLE.

#########################################

# Percentage Error = ( ( True Value – Predicted Value ) / True Value ) * 100 
# Same can be converted to mean percentage error for all samples. 

def mean_percentage_error(y_true, y_pred): 
    """ 
    This function calculates mpe 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: mean percentage error 
    """ 
    # initialize error at 0 
    error = 0 
    # loop over all samples in true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate percentage error  
        # and add to error 
        error += (yt - yp) / yt 
    # return mean percentage error 
    return error / len(y_true) 

####################################################

# If we take the absolute of the percentages in the above equation, it becomes Mean Absolute Percentage Error!

def mean_abs_percentage_error(y_true, y_pred): 
    """ 
    This function calculates MAPE 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: mean absolute percentage error 
    """ 
    # initialize error at 0 
    error = 0 
    # loop over all samples in true and predicted list 
    for yt, yp in zip(y_true, y_pred): 
        # calculate percentage error  
        # and add to error 
        error += np.abs(yt - yp) / yt 
    # return mean percentage error 
    return error / len(y_true) 

##################################################

# R-Squared metric compares your model to the base model which predicts y_true_mean all the time.
# If your model hs higher error as compared to the base model, then R2 will be close to 0

def r2(y_true, y_pred): 
    """ 
    This function calculates r-squared score 
    :param y_true: list of real numbers, true values 
    :param y_pred: list of real numbers, predicted values 
    :return: r2 score 
    """ 
    # calculate the mean value of true values 
    mean_true_value = np.mean(y_true) 
    # initialize numerator with 0 
    numerator = 0 
    # initialize denominator with 0 
    denominator = 0 
    # loop over all true and predicted values 
    for yt, yp in zip(y_true, y_pred): 
        # update numerator 
        numerator += (yt - yp) ** 2 
        # update denominator 
        denominator += (yt - mean_true_value) ** 2 
        # calculate the ratio 
        ratio = numerator / denominator 
    # return 1 - ratio 
    return 1 - ratio 

