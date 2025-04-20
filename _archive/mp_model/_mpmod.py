import os
import warnings
import urllib3
import pickle

import polars as pl
import pandas as pd

from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pl.MapWithoutReturnDtypeWarning)
# warnings.filterwarnings("ignore", category=urllib3.InsecureRequestWarning)
urllib3.disable_warnings()

from mp_model import data, correction, classification, identification, evaluation, preprocess

from mp_model._utils import (
    DefaultLogger,
)
import mp_model._save_utils as save_utils

logger = DefaultLogger()
logger.configure("INFO")

class MpModel:
    def __init__(self,
                 file_input:str=None, file_output:str=None,
                 dir_input:str=None, dir_output:str=None,
            
                 verbose: bool=False,) -> None:

        self.file_input = file_input
        self.file_output = file_output

        self.dir_input = dir_input
        self.dir_output = dir_output

        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')

    def set_data(self) -> None:
        # Setting input data
        if self.file_input and self.dir_input:
            logger.info(f"Both file_input and dir_input has been provided.\nOnly processing {self.dir_input}.")
            self.data_path = self.dir_input
        elif not self.file_input and not self.dir_input:
            logger.error(f"Both file_input and dir_input are empty.\nAborting.")
            raise ValueError(f"Both file_input and dir_input are empty."
                             f"Aborting.")
        else:
            self.data_path = (self.file_input | self.dir_input)
            logger.info(f"Processing {self.data_path}.")

        # Check if inputted data path exists
        if not data.check_exist(path_file=self.data_path):
            logger.error(f"Inputted file or directory {self.data_path} does not exist.\nAborting.")
            raise ValueError(f"Inputted file or directory {self.data_path} does not exist."
                             f"Aborting.")
        
        data_len, self.data = data.load_utils(self.data_path)
        if data_len == 0:
            logger.error(f"Failed to load file(s) due to it being of unloadable type.\nAborting.")
            raise ValueError(f"Failed to load file(s) due to it being of unloadable type."
                             f"Aborting.")
        
        # Setting output data
        if data_len == 1 and self.file_output and self.dir_output:
            self.output_is_file = True
            self.output_path = os.path.join(self.dir_output, self.file_output)
            logger.info(f"Setting output path to {self.output_path}")
        else:
            self.output_is_file = False
            if self.dir_output:
                self.output_path = self.dir_output
                logger.info(f"Setting output directory to {self.output_path}")

            else:
                dir_output = './output'
                logger.info(f"Setting output directory to {dir_output}")

            if data.check_directory_path(self.output_path) == 0:
                logger.info(f"Created output directory: {self.output_path}")

    def ocr_correction(self) -> None:
        # TODO: run preprocessing
        
        if self.ocr_method == 'bart_basic':
            self.data = correction.ocr_bart_basic(self.data)

    def run_identification(self,
                           method: str='er') -> None:
        if method == 'er':
            identification.run_er()

        elif method == 'fuzzy':
            identification.run_fuzzy()
    
    def run_classification(self) -> None:

        return 0
    
    def run_evaluation(self) -> None:
        # evaluation.result_table()
        pass
    
    def retrain_model(self,
                      training_data:str) -> None:
        
        model_directory = './mp_model/_model/'

        
        pass

    def save_data(self) -> None:
        data.save_data(dict_data=self.data,
                       bool_output_file=self.output_is_file,
                       path_output=self.output_path, save_format='JSON')