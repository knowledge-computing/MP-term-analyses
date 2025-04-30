from typing import Union, List, Dict, Tuple

from transformers import pipeline
from transformers import BertTokenizerFast, DistilBertTokenizerFast 
from transformers import AutoModelForTokenClassification 

def load_model(model_path:str):
    """

    Parameters
    : model_path (str): absolute path to the trained model

    Return
    : pipeline (obj):
    """
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    ner_pipeline = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, aggregation_strategy="simple", device=0)

    return ner_pipeline

def run_nermodel(ner_pipeline,
                 input_sentence:Union[str, List[str]]) -> Union[str, List[str]]:
    
    list_output = {}
    if not isinstance(input_sentence, list):
        list_input = [input_sentence]
    else:
        list_input = input_sentence

    for res in list_input:
        ner_results = ner_pipeline(res)
        list_output[res] = ner_results
    
    return list_output