import os
import logging
from typing import List, Dict
from datetime import datetime
import json

import polars as pl
from mpterm import data

class DefaultLogger:
    def __init__(self):
        self.logger = logging.getLogger('MappingPrejudice')
        self.log_dir = './logs/'

    def configure(self, 
                  level):
        self.set_level(level)
        self._add_handler()

    def set_level(self, 
                  level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def info(self, 
             message:str):
        self.logger.info(message)

    def warning(self, 
                message:str):
        self.logger.warning(message)
        
    def error(self, 
              message:str):
        self.logger.error(message)

    def _add_handler(self):
        # Initiating streamhandler i.e., terminal
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        self.logger.addHandler(sh)

        # Initiating filehandler i.e., log file
        data.check_directory_path(path_directory='./logs/')  # Creating log folder if not exist
        fh = logging.FileHandler(f'./logs/mp_er_{datetime.timestamp(datetime.now())}.log')
        fh.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
        self.logger.addHandler(fh)

def str_to_json_extracted(input_unk_info):
    """

    """
    data_path = 'UNK'
    uuid = 'NONE'
    if isinstance(input_unk_info, str):
        input_unk_info = json.loads(input_unk_info)

    if isinstance(input_unk_info, dict):
        try: 
            data_path = input_unk_info['ocr_json']
            uuid = input_unk_info['uuid']
        except: pass
    else:
        data_path = input_unk_info

    return data_path, uuid