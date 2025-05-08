from typing import Union, List

import torch
from transformers import pipeline
from transformers import DistilBertTokenizerFast 
from transformers import AutoModelForTokenClassification 

def load_model(model_path:str):
    """
    Initiates the entity recognition pipeline
    
    Parameters
    : model_path (str): absolute path to the trained model

    Return
    : pipeline (obj): NER pipeline object that can be used for entity recognition
    """
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    # Defaulting to use CUDA with GPU support if available, else run on CPU
    if torch.cuda.is_available():
        device_num = 0
    else: device_num = -1

    # Initiating ner pipeline
    ner_pipeline = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, aggregation_strategy="simple", device=device_num)

    return ner_pipeline

def run_nermodel(ner_pipeline,
                 input_sentence:Union[str, List[str]]) -> Union[List[str]]:
    """
    Runs entity recognitition using the loaded pipeline

    Parameters
    : ner_pipeline (obj): NER pipeline object that can be used for entity recognition
    : input_sentence (str | List(str)): sentence(s) to run ER on

    Output
    : list_output (list(str)): 
    """
    
    list_output = {}
    if not isinstance(input_sentence, list):
        list_input = [input_sentence]
    else:
        list_input = input_sentence

    for res in list_input:
        ner_results = ner_pipeline(res)
        list_output[res] = ner_results
    
    return list_output