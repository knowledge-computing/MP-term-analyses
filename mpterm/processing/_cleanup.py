import os
import json
from typing import Union, List, Dict, Tuple

import re

def get_components(path_file:str) -> Tuple[str]:
    """
    
    """
    list_components = path_file.split('/')
    workflow = ''

    for i in list_components:
        if '-county' in i:
            workflow = i

    lookup = path_file.split(workflow)[1]
    lookup, _ = os.path.splitext(lookup)

    updated_path_data = os.path.join(f'ocr/txt/{workflow}/{lookup}.txt')

    return updated_path_data, workflow, lookup

def get_line_num(list_sentences:List[str], 
                 list_lines:List[str]):
    """
    Identify start of sentence (SOS) and end of sentence (EOS) line number
    """
    dict_line_num = {}

    sentence_idx = 0
    munched_sent = list_sentences[sentence_idx]

    num_SOS = 1
    num_EOS = 0
    for idx, line in enumerate(list_lines):
        if line.strip() in munched_sent:
            munched_sent = re.sub(line, '', munched_sent)
            munched_sent = munched_sent.strip()

        else:
            # Add result to dictionary of line numbers
            dict_line_num[list_sentences[sentence_idx]] = [num_SOS, idx+1]

            # Move on to next sentence
            sentence_idx += 1 
            munched_sent = list_sentences[sentence_idx]

            num_SOS = num_EOS
            if munched_sent not in line.strip():
                # Case the sentence end = line end
                num_SOS += 1

    return dict_line_num

def format_output(dict_ners:Dict[str, List[str]], 
                  dict_line_num: Dict[str, List[int]], list_lines: List[str],
                  dict_info: Dict[str, str],) -> dict:
    """
    Formats the output such that it matches the Zooniverse output format:
    Expected: {term: [lines], workflow: str, lookup: str, uuid: str}
    """
    zooniverse_output = {}

    # Returning true line number of the matched term
    for sentence, list_ners in dict_ners.items():
        num_SOS, num_EOS = dict_line_num[sentence]
        cur_idx = num_SOS

        for ner in list_ners:
            if ner in list_lines[cur_idx-1]:
                try: zooniverse_output[ner].append(cur_idx)
                except: zooniverse_output[ner] = [cur_idx]

    # Appending the defaults
    zooniverse_output.update(dict_info)

    return zooniverse_output