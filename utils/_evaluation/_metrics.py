from Typing import List
from sklearn.metrics import precision_score, recall_score

def report_recall(list_true:list, list_pred:list) -> float:
    """
    
    """

    return float(recall_score(y_true=list_true, y_pred=list_pred, zero_division=0) * 100)

def report_precision(list_true:list, list_pred:list) -> float:
    """

    """
    return float(precicion_score(y_true=list_true, y_pred=list_pred, zero_division=0) * 100)

def report_perplexity() -> float:
    """
    Used to measure the amount of bias in each language model


    """
    return 0