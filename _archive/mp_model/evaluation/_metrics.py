from Typing import List
from sklearn.metrics import precision_score, recall_score

def report_recall(list_true:List[int], list_pred:List[int],
                  bool_percentage:bool=True) -> float:
    """
    Returns the recall score of model

    Parameters
    : list_true (List[int]): list of ground truth classes
    : list_pred (List[int]): list of predicted classes
    : bool_percentage (bool=True): true if return as percentage; false if return as decimal

    Return
    : recall_value (float): recall score 
    """
    recall_value = recall_score(y_true=list_true, y_pred=list_pred, zero_division=0)

    if not bool_percentage:
        return recall_value
    
    return float(recall_value * 100)

def report_precision(list_true:List[int], list_pred:List[int],
                     bool_percentage:bool=True) -> float:
    """
    Returns the precision score of model

    Parameters
    : list_true (List[int]): list of ground truth classes
    : list_pred (List[int]): list of predicted classes
    : bool_percentage (bool=True): true if return as percentage; false if return as decimal

    Return
    : precision_value (float): precision score 
    """

    precision_value = precision_score(y_true=list_true, y_pred=list_pred, zero_division=0)

    if not bool_percentage:
        return precision_value
    
    return float(precision_value * 100)

def report_perplexity() -> float:
    """
    Used to measure the amount of bias in each language model

    TODO


    """
    return 0

def report_tscore(alpha:float=0.05) -> float:
    """
    Used to measure the amount of bias in each language model

    TODO


    """
    return 0

def report_anova(alpha:float=0.05) -> float:
    """
    Analysis of variance (ANOVA): compare the means of two or more groups by analyzing variance
    Used in: 

    Equation: 

    TODO


    """
    return 0