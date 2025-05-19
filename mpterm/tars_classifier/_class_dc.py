from typing import List, Dict

def determine_document(tars_result:dict) -> bool:
    """
    Outputs whether the document contains racial restriction as classified by TARS

    Parameters
    : tars_resulst (dict): final TARS label from the model

    Return
    : (bool): indication if document contains racial restriction or not
    """
    
    for _, label in tars_result.items():
        if label == 'RACE':
            return True

    return False