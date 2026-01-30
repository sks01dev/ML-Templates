"""
QWK/ Cohen's Kappa measures the “agreement” between two “ratings”. The ratings can be any real numbers in 0 to N. 
And predictions are also in the same range. An agreement can be defined as how close 
these ratings are to each other. So, it’s suitable for a classification problem with N 
different categories/classes. If the agreement is high, the score is closer towards 1.0. 
In the case of low agreement, the score is close to 0.
"""
metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic") metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic") 



#####################################################################################
"""
An important metric is Matthew’s Correlation Coefficient (MCC). MCC ranges 
from -1 to 1. 1 is perfect prediction, -1 is imperfect prediction, and 0 is random 
prediction. The formula for MCC is quite simple. 

MCC = (TP * TN - FP * FN) /  [ (TP + FP) * (FN + TN) * (FP + TN) * (TP + FN) ] ^ (0.5)

We see that MCC takes into consideration TP, FP, TN and FN and thus can be used 
for problems where classes are skewed.
"""

def mcc(y_true, y_pred): 
  """ 
  This function calculates Matthew's Correlation Coefficient 
  for binary classification. 
  :param y_true: list of true values 
  :param y_pred: list of predicted values 
  :return: mcc score 
  """ 
  tp = true_positive(y_true, y_pred) 
  tn = true_negative(y_true, y_pred) 
  fp = false_positive(y_true, y_pred) 
  fn = false_negative(y_true, y_pred) 
  numerator = (tp * tn) - (fp * fn) 
  denominator = ( 
    (tp + fp) * 
    (fn + tn) * 
    (fp + tn) * 
    (tp + fn) 
  ) 
  denominator = denominator ** 0.5 
  return numerator/denominator 


