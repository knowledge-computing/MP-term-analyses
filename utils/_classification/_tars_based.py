from typing import List
from flair.models import TARSClassifier
from flair.data import Sentence

def tars_classifier_zero(list_sentences:List[str]|str,
                         classes:List[str]|str,
                         model:str='tars-base') -> List[str]|str:
                         
    tars = TARSClassifier.load(model)

    if isinstance(classes, str):
        classes = [classes]

    if isinstance(list_sentences, str):
        sentence = Sentence(list_sentences)
        tars.predict_zero_shot(sentence, classes)
        return sentence
    
    list_identified_sentences = []
    for line in list_sentences:
        sentence = Sentence(line)
        tars.predict_zero_shot(sentence, classes)
        list_identified_sentences.append(sentence)

    return list_identified_sentences