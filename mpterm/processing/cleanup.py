import os
import json
from typing import Union, List, Dict, Tuple

def format_output(dict_ners:Dict[str, List[int]], list_org_lines:List[str],
                  dict_info: Dict[str, str],) -> dict:
    """
    Formats the output such that it matches the Zooniverse output format:
    Expected: {term: [lines], workflow: str, lookup: str, uuid: str}
    """

    ner_idx = 0
    key, value = dict_ners.items()[ner_idx]

    zooniverse_output = {}
    for idx, line in enumerate(list_org_lines):
        if key in line:
            try: zooniverse_output[value].append(idx)
            except: zooniverse_output[value] = [idx]

            ner_idx += 1
            key, value = dict_ners.items()[ner_idx]

    # Appending the defaults
    zooniverse_output.update(dict_info)

    return zooniverse_output