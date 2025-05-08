from typing import Union, List, Dict, Tuple

import re
import spacy
spacy_model = spacy.load("en_core_web_sm")

def to_sentence(input_strs:Union[str, List[str]]) -> List[str]:
    """
    Converts original document in form of list and perform sentence tokenization

    Parameter
    : input_strs (str | List[str]): original input

    Return
    : list_newsentence (List[str]): sentence-ized input
    """
    text_block = ""
    if isinstance(input_strs, list):
        text_block = " ".join(input_strs)
    else:
        text_block = input_strs

    text_block = re.sub('\n', ' ', text_block)

    spacy_doc = spacy_model(text_block)

    list_newsentence = []
    for sent in spacy_doc.sents:
        list_newsentence.append(str(sent).strip())
    
    return list_newsentence