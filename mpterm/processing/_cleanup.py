import os
from typing import List, Dict, Tuple

def get_components(path_file:str) -> Tuple[str]:
    """
    Returns data information

    Parameter
    : list_sentences (List[str]): sentence-ized input
    : list_lines (List[str]): original input

    Return
    : updated_path_data (str): updated path to ocr'd text
    : workflow (str): county information 
    : lookup (str): lookup (remaining path to file)
    """
    list_components = path_file.split('/')
    workflow = ''

    for i in list_components:
        if '-county' in i:
            workflow = i

    lookup = path_file.split(workflow)[1].strip('/')
    lookup, _ = os.path.splitext(lookup)

    updated_path_data = os.path.join(f'ocr/txt/{workflow}/{lookup}.txt')

    return updated_path_data, workflow, lookup

def get_line_num(list_sentences:List[str], 
                 list_lines:List[str]) -> Dict[str, List[int]]:
    """
    Identify start of sentence (SOS) and end of sentence (EOS) line number

    Parameter
    : list_sentences (List[str]): sentence-ized input
    : list_lines (List[str]): original input

    Return
    : dict_line_num (Dict[str, List[int]]): dictionary giving starting and ending line num of each sentence
    """
    dict_line_num = {}

    sentence_idx = 0
    dict_line_num = {list_sentences[sentence_idx]: [1, 0]}
    munched_sent = list_sentences[sentence_idx]
    for idx, line in enumerate(list_lines):
        bool_line_in_sent = False

        while not bool_line_in_sent:
            if line.strip() in munched_sent:
                # Remove the current line portion from the sentence
                munched_sent = munched_sent.split(line.strip(), maxsplit=1)[1]
                munched_sent = munched_sent.strip()

                # Update the EOS index of the sentence
                dict_line_num[list_sentences[sentence_idx]][1] = idx + 1

                if len(munched_sent) == 0:
                    # Proceed to next sentence in list
                    sentence_idx += 1

                    if sentence_idx < len(list_sentences):
                        # Initiate with after index and 0 for the next sentence
                        dict_line_num[list_sentences[sentence_idx]] = [idx + 2, 0]
                        munched_sent = list_sentences[sentence_idx]

                bool_line_in_sent = True

            else:
                # Update EOS to end of the sentence
                dict_line_num[list_sentences[sentence_idx]][1] = idx + 1

                # Remove remaining sentence part from the line
                line = line.split(munched_sent.strip(), maxsplit=1)[1]
                line = line.strip()

                # Start the next sentence and initialize it
                sentence_idx += 1
                dict_line_num[list_sentences[sentence_idx]] = [idx + 1, 0]
                munched_sent = list_sentences[sentence_idx]

                bool_line_in_sent = False

    return dict_line_num

def format_output(dict_ners:Dict[str, List[str]], 
                  dict_line_num: Dict[str, List[int]], list_lines: List[str],
                  dict_info: Dict[str, str],) -> dict:
    """
    Formats the output such that it matches the Zooniverse output format:
    Expected: {term: [lines], workflow: str, lookup: str, uuid: str}

    Parameters
    : dict_ners (Dict[str, List[str]]): racial entities per each sentence
    : dict_line_num (Dict[str, List[int]]): sentence mapped to start and end line number
    : list_lines (List[str]): Original document in string format
    : dict_info (Dict[str, str]): workflow, lookup, and uuid information

    Return
    zooniverse_output (dict): entity formatted in form of zooniverse output
    """
    zooniverse_output = {}
    bool_entity_recognized = False

    # Returning true line number of the matched term
    for sentence, list_ners in dict_ners.items():
        if len(list_ners) > 0:
            # Indicating whether there is identified entity
            bool_entity_recognized = True

            num_SOS, num_EOS = dict_line_num[sentence]
            cur_idx = num_SOS

            for cur_idx in list(range(num_SOS, num_EOS+1)):
                for ner in list_ners:
                    if ner in list_lines[cur_idx-1]:
                        try: zooniverse_output[ner].append(cur_idx)
                        except: zooniverse_output[ner] = [cur_idx]

    if bool_entity_recognized:
        # Appending the defaults
        zooniverse_output.update(dict_info)

        return zooniverse_output
    
    else:
        return {}