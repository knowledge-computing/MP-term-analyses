import os
import warnings
import urllib3
import pickle

import polars as pl
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pl.MapWithoutReturnDtypeWarning)
# warnings.filterwarnings("ignore", category=urllib3.InsecureRequestWarning)
urllib3.disable_warnings()

from mp_model_train import data

from mp_model_train._utils import (
    DefaultLogger,
)

logger = DefaultLogger()
logger.configure("INFO")

class MpModelTrainer:
    def __init__(self,
                 file_ground_truth:str=None,
                 dir_txt_files:str=None,
            
                 verbose: bool=False,) -> None:

        self.file_gt = file_ground_truth
        self.dir_txt_files = dir_txt_files

        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')

    def format_gt(self,) -> None:
        folder_cov_path = os.path.join(self.dir_txt_files, data.get_cov_path(self.file_gt))

        if not data.check_exist(folder_cov_path):
            logger.error(f"Data path does not exist: {folder_cov_path}.\nAborting")
            raise ValueError(f"Data path does not exist: {folder_cov_path}."
                             f"Aborting")

        # Load ground truth file
        self.gt_data = data.load_data(self.file_ground_truth).select(
            pl.col(['cov_text']),
            pl.col('image_ids').str.split(',')
        ).explode('image_ids').drop_nulls('image_ids')
        
        # Get absolute data path
        self.gt_data = self.gt_data.with_columns(
            pl.concat_str([
                pl.lit(folder_cov_path),
                pl.col('image_ids')
            ]).alias('data_path')
        ).drop('image_ids')
        
        # Check whether indicated file path exists on local path
        self.gt_data = self.gt_data.filter(
            pl.col('data_path').map_elements(lambda x: data.check_exist(x), return_dtype=bool)
        )
        logger.info(f"Found {self.gt_data.shape[0]} files listed in ground truth file {self.file_gt}.")

        # Load the actual content of the files
        self.gt_data = self.gt_data.with_columns(
            actual_data = pl.col('data_path').map_elements(lambda x: load_data(x), return_dtype=str)
        ).filter(
            pl.col('actual_data').str.contains(pl.col('cov_text'))
        )