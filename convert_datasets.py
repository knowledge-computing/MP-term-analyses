import os
import polars as pl
import pandas as pd
import pickle
import json
from datasets import load_dataset, Dataset
from transformers.pipelines.pt_utils import KeyDataset
from collections import OrderedDict

from typing import List, Dict

import warnings
from polars.exceptions import MapWithoutReturnDtypeWarning
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-large-ocr')
tokenizer = AutoTokenizer.from_pretrained('pykale/bart-large-ocr')

generator = pipeline('text2text-generation', 
                     model=model.to('cuda'), 
                     tokenizer=tokenizer, 
                     device='cuda', 
                     max_length=1024)

def load_org_data(path_file:str,
                  bool_both_forms:bool=False) -> List[str]:
    f = open(path_file, 'r')
    list_lines = f.readlines()
    f.close()

    paragraph = "\n".join(list_lines)
    paragraph = paragraph.replace("-\n", "").replace("\n", " ")

    return {'list_lines': list_lines, 'paragraph': paragraph}

def strdict2listtup(list_str:List[str]) -> str:
    """
    
    """
    loaded_dict = {}

    for i in list_str:
        if not i:
            continue

        tmp_dict = json.loads(i)
        tmp_dict.pop('workflow')
        tmp_dict.pop('lookup')
        tmp_dict.pop('uuid')

        loaded_dict.update(tmp_dict)

    if len(loaded_dict) == 0:
        return []
    
    dict_lines = {}
    for k, v in loaded_dict.items():
        for i in v:
            try:
                dict_lines[str(i)].append(k.strip())
            except:
                dict_lines[str(i)] = [k.strip()]

    dict_lines = OrderedDict(sorted(dict_lines.items()))

    return json.dumps(dict_lines)

def temp(list_lines: List[str],
         contents_to_find:str):
    data = {
        'TD': list(range(len(list_lines))),
        'text': list_lines
    }
    dataset = Dataset.from_dict(data)
    ocr_corrected_ds = generator(KeyDataset(dataset, "text"), batch_size=32)
    print(list(ocr_corrected_ds))

    # if contents_to_find:
    #     dict_finds = json.loads(contents_to_find)

    #     for line_num, term in dict_finds.items():
    #         ocr_corrected_line = generator(list_lines[int(line_num)])[0]['generated_text']

    #         print(ocr_corrected_line, term)

    return 0

# def create_dataset():
#     dataset = load_dataset('csv', data_files={'train': 'train.txt', 'validation': 'val.txt', 'test': 'test.txt'}, sep=",", 
#                                 names=["text", "tokens", "ner_tags"])
#     labels = df['label'].unique().tolist()
#     ClassLabels = ClassLabel(num_classes=len(labels), names=labels)
    

#     dataset.class_encode_column('ner_tags')

#     return 0

dir_splitted_data = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/splitted_data/mn-anoka'
dir_gt_data = '/home/yaoyi/pyo00005/Mapping_Prejudice/data'

for f in os.listdir(dir_splitted_data):
    file_path = os.path.join(dir_splitted_data, f)
    pl_data = pl.read_csv(file_path).select(
        pl.col(['cov_confirmed', 'hit_contents_basic', 'hit_contents_fuzzy']),
        pl.col('page_ocr_text').map_elements(lambda x: os.path.join(dir_gt_data, x))
    ).drop_nulls('cov_confirmed')

    pl_data = pl_data.with_columns(
        tmp = pl.col('page_ocr_text').map_elements(lambda x: load_org_data(path_file=x, bool_both_forms=True)),
        tmp_contents = pl.struct(pl.all()).map_elements(lambda x: strdict2listtup([x['hit_contents_basic'], x['hit_contents_fuzzy']]))
    ).unnest('tmp').with_columns(
        pl.when(pl.col('cov_confirmed') == True)
        .then(pl.col('tmp_contents'))
        .otherwise(pl.lit(None)).alias('hit_contents')
    ).drop(['hit_contents_basic', 'hit_contents_fuzzy', 'tmp_contents'])

    pl_data = pl_data.with_columns(
        identified_components = pl.struct(pl.all()).map_elements(lambda x: temp(x['list_lines'], x['hit_contents']))
    )

    break