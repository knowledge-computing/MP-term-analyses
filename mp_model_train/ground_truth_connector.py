import os
import re
import polars as pl

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

def tmp_connect(file_ground_truth:str,
                dir_txt_files:str):
    folder_cov_path = os.path.join(dir_txt_files, get_cov_path(file_ground_truth))

    if not check_exist(folder_cov_path):
        raise ValueError(f"Path does not exist")
    
    pl_gt = load_data(file_ground_truth).select(
        pl.col(['cov_text']),
        pl.col('image_ids').str.split(',')
    ).explode('image_ids').drop_nulls('image_ids').with_columns(
        pl.concat_str([
            pl.lit(folder_cov_path),
            pl.col('image_ids')
        ]).alias('data_path')
    ).drop('image_ids').filter(
        pl.col('data_path').map_elements(lambda x: check_exist(x), return_dtype=bool)
    )


    # TODO: check if the following works
    pl_gt = pl_gt.with_columns(
        actual_data = pl.col('data_path').map_elements(lambda x: load_data(x), return_dtype=str)
    ).filter(
        pl.col('actual_data').str.contains(pl.col('cov_text'))
    )

    print(pl_gt)
    
    

tmp_connect(ground_truth, dir_txt_files)
# print(pl_data)