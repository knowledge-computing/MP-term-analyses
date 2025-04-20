import os
import json
import warnings
from typing import Dict, List, Union

import math
import pandas as pd
import polars as pl
from nltk.tokenize import word_tokenize

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

    return json.dumps(dict_lines)

def loadtextfile(path_dir:str, path_file:str) -> List[str]:
    path_full = os.path.join(path_dir, path_file)

    f = open(path_full, 'r')
    list_lines = f.readlines()
    f.close()

    return list_lines

def split_data(df_data:Union[pl.DataFrame, pd.DataFrame],
               split_ratio:Dict[str, float],
               dir_output:str, file_name:str, class_col:str=None, save_splitted:bool=False) -> Union[pl.DataFrame, pd.DataFrame]:
    
    len_data = df_data.shape[0]

    list_split = []

    for idx, (tag, ratio) in enumerate(split_ratio.items()):
        if idx == len(split_ratio) - 1:
            remaining_count = len_data - len(list_split)
            list_split.extend([tag] * remaining_count)
            break

        count = math.ceil(ratio * len_data)
        list_split.extend([tag] * count)

    if isinstance(df_data, pl.DataFrame):
        df_data = df_data.sample(fraction=1, shuffle=True)
        df_data = df_data.with_columns(
            data_split = pl.Series(list_split)
        )

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    file_name = os.path.join(dir_output, file_name)
    if not save_splitted:
        df_data.write_csv(f"{file_name}.csv")
    
    if save_splitted:
        list_df_data = df_data.partition_by('data_split')
        for df in list_df_data:
            partition_name = df.item(0, 'data_split')
            df.write_csv(f'{file_name}_{partition_name}.csv')

    return df_data

def toknlabel(full_text: List[str],
              hit_contents = str) -> Dict[str, List[str]]:
    
    hit_contents = json.loads(hit_contents)

    all_tokens = []
    for idx, line in enumerate(full_text):
        all_tokens.extend(word_tokenize(line))
    
    
    # print(all_tokens)
    return{'tokenized_text':all_tokens, 'labels':[]}

FILE_NAME = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/labeled_data/mn-anoka-county_deedpage_sample_post_zooniverse_mn-anoka-county_100pct_20250410_1706.csv'
PATH_RAW_DATA = '/home/yaoyi/pyo00005/Mapping_Prejudice/data'

pl_data = pl.read_csv(FILE_NAME, infer_schema_length=0)
pl_data = pl_data.select(
    pl.col(['page_ocr_text', 'cov_confirmed', 'hit_contents_basic', 'hit_contents_fuzzy']),
).filter(pl.col('cov_confirmed') != 'None').with_columns(
    hit_contents = pl.struct(pl.all()).map_elements(lambda x: strdict2listtup([x['hit_contents_basic'], x['hit_contents_fuzzy']])),
    full_text = pl.col('page_ocr_text').map_elements(lambda x: loadtextfile(PATH_RAW_DATA, x))
)

dict_split = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
pl_data = split_data(pl_data, dict_split, class_col='cov_confirmed', dir_output='/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/zooniverse/splitted_data/', file_name='mn-anoka')
# .with_columns(
#     tmp = pl.struct(pl.all()).map_elements(lambda x: toknlabel(x['full_text'], x['hit_contents']))
# )

# print(pl_data)