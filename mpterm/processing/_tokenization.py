import os
from typing import Union, List, Dict, Tuple

import spacy
spacy_model = spacy.load("en_core_web_sm")

def to_sentence(input_strs:str|List[str],
                bool_ocr_correct:bool = False) -> List[str]:
    text_block = ""
    if isinstance(input_strs, list):
        text_block = " ".join(input_strs)
    else:
        text_block = input_strs
    
    spacy_doc = spacy_model(text_block)

    list_newsentence = []
    for sent in spacy_doc.sents:
        list_newsentence.append(str(sent))

    if bool_ocr_correct:
        pass

    return list_newsentence

def to_tokens() -> List[str]:
    return 0