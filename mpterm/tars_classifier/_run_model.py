from typing import Union, List

from flair.data import Sentence
from flair.models import TARSClassifier

def load_model(model_path:str):
    """
    Initiates the document classification pipeline
    
    Parameters
    : model_path (str): absolute path to the trained model

    Return
    : pipeline (obj): TARS pipeline object that can be used for document classification
    """
    tars_pipeline = TARSClassifier.load(model_path)

    # Switch to trained task
    tars_pipeline.switch_to_task("question classification")

    return tars_pipeline

def run_tarsmodel(tars_pipeline,
                  input_sentence:Union[str, List[str]]) -> Union[List[str]]:
    """
    Runs document classification using the loaded pipeline

    Parameters
    : tars_pipeline (obj): TARS pipeline object that can be used for document classification
    : input_sentence (str | List(str)): sentence(s) to run DC on

    Output
    : list_output (list(str)): 
    """
    
    list_output = {}
    if not isinstance(input_sentence, list):
        list_input = [input_sentence]
    else:
        list_input = input_sentence

    for res in list_input:
        # Convert each sentence into a Flair Sentence Object
        sentence = Sentence(res)
        tars_pipeline.predict(sentence)

        # Run TARS pipeline
        list_output[res] = str(sentence.get_label().value)
    
    return list_output