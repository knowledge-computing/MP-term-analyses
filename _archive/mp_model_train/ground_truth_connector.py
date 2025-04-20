import os
import re
import polars as pl
from typing import List, Dict, Tuple

ground_truth = '/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/covenants-mn-anoka-county.csv'
dir_txt_files = '/home/yaoyi/pyo00005/Mapping_Prejudice/data/ocr/txt'

def get_basename(filename:str) -> str:
    file_name, _ = os.path.splitext(os.path.basename(filename))

    return file_name

def get_cov_path(filename:str) -> str:
    base_filename = get_basename(filename)    
    cov_region = re.search(r'(?<=covenants-).+', base_filename).group(0)

    return cov_region

def check_exist(data_path:str) -> bool:
    """
    Checks whether the inputted file (data_path) exists

    Paremters
    : data_path (str): 

    Return
    TODO
    """
    if os.path.exists(data_path):
        return True
    
    return False

def return_basename(data_path: str) -> str:
    """
    Returns the file name (i.e., the portion before the extension)

    Parameters
    : data_path (str): 

    Return
    TODO
    """
    file_name, _ = os.path.splitext(os.path.basename(data_path))

    return file_name

def check_mode(data_path: str) -> str | int:
    """
    Returns the output type (extension or dir)

    Parameters
    : data_path (str):

    Return
    TODO
    """
    _, file_extension = os.path.splitext(data_path)

    if file_extension:
        return file_extension
    
    return -1

def load_data(data_path:str) -> pl.DataFrame | str:
    """
    Load data into a dictionary with key as file name and value as the document string

    Parameters
    : data_path (str): 

    Return
    : TODO
    """
    datamode = check_mode(data_path)
    if datamode == -1:
        raise ValueError("of unacceptable type")
    
    if datamode == '.txt':
        with open(data_path, 'r') as f:
            content = f.read()

        return content

    if datamode == '.csv':
        pl_data = pl.read_csv(data_path)

        return pl_data

def pick_out_terms(input_text:str, regex_pattern:str) -> Tuple[str, Dict[str, List[Tuple[int]]]]:
    """
    Tuple[input_text, Dict[term, List[Tuple[str_start, str_end]]]]
    """

    re_pattern = re.compile(regex_pattern)
    pos = 0
    out = {}

    while m := re_pattern.search(input_text, pos):
        try:
            out[m.group()].append(m.span())
        except:
            out[m.group()] = [m.span()]

        pos = m.span()[1]

    return out

def tmp_connect(file_ground_truth:str,
                dir_txt_files:str):
    folder_cov_path = os.path.join(dir_txt_files, get_cov_path(file_ground_truth)) + '/'

    if not check_exist(folder_cov_path):
        raise ValueError(f"Path does not exist")
    
    pl_gt = load_data(file_ground_truth).select(
        pl.col(['cov_text']),
        pl.col('image_ids').str.split(',')
    ).explode('image_ids').drop_nulls('image_ids').with_columns(
        pl.concat_str([
            pl.lit(folder_cov_path),
            pl.col('image_ids'),
            pl.lit('.txt')
        ]).alias('data_path')
    ).drop('image_ids').filter(
        pl.col('data_path').map_elements(lambda x: check_exist(x), return_dtype=bool)
    )

    pl_gt = pl_gt.with_columns(
        actual_data = pl.col('data_path').map_elements(lambda x: load_data(x), return_dtype=str),
        file_name = pl.col('data_path').str.split('/').list.get(-1).str.split('_SPLITPAGE_').list.first().str.split('.txt').list.first()
    )

    pl_term_list_scaling = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/mp_model/_seed_terms/scaling-ai.csv').with_columns(
        pl.col('term').str.to_lowercase())['term'].to_list()
    pl_term_list_mp_filtered = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/mp_model/_seed_terms/mp_reduced.csv').with_columns(
        pl.col('term').str.to_lowercase())['term'].to_list()

    pl_term_list_scaling.extend(pl_term_list_mp_filtered)
    unique_seed_terms = list(set(pl_term_list_scaling))

    regex_list = '|'.join(unique_seed_terms)

    # print(pl_gt.filter(~pl.col('actual_data').str.contains(pl.col('cov_text'), literal=True)).item(0, 'data_path'))

    pl_gt = pl_gt.group_by('file_name').agg([pl.all()]).with_columns(
        pl.col('cov_text').list.first(),
        pl.col('actual_data').list.join("\n")
    )

    pl_gt = pl_gt.with_columns(
        term_exist = pl.col('cov_text').str.to_lowercase().map_elements(lambda x: pick_out_terms(x, regex_list))
    )


    print(pl_gt)
    

# tmp_connect(ground_truth, dir_txt_files)


pl_tmp = pl.read_csv('/home/yaoyi/pyo00005/Mapping_Prejudice/ground_truth/mn-washington-county_deedpage_sample_post_zooniverse_mn-washington-county_100pct_20250410_1124.csv')
pl_tmp = pl_tmp.drop_nulls('cov_confirmed').select(
    pl.col(['cov_confirmed', 'hit_contents_basic', 'hit_contents_fuzzy', 'page_ocr_text'])
)

print(pl_tmp)