import os
import warnings
import polars as pl
from polars.exceptions import MapWithoutReturnDtypeWarning
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)

from mpterm import data, processing, entity_recognizer

from mpterm._utils import (
    DefaultLogger,
)

logger = DefaultLogger()
logger.configure("INFO")

class MPTerm:
    def __init__(self,
                 path_data: str,
                 dir_output:str=None,
                 file_output:str=None,
                 verbose: bool=False) -> None:
        
        self.path_data = path_data
        self.dir_output = dir_output
        self.file_output = file_output
        self.bool_ocr_correct = False

        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')

    def load_data(self):
        int_existence = data.check_exists(self.path_data)
        if int_existence == -1:
            logger.error(f"File {self.path_data} does not exist")
            raise ValueError(f"File {self.path_data} does not exist",
                              "Ending program")

        self.list_lines = data.load_data(self.path_data)
        logger.info(f"File {self.path_data} loaded\nTotal of {len(self.list_lines)} lines")

        self.list_sentences = processing.to_sentence(input_strs=self.list_lines, 
                                                        bool_ocr_correct=self.bool_ocr_correct)
        
        # TODO: complete getting the data_info
        self.data_info = {}

    def entity_recog(self, ner_model_path:str=None):
        if not ner_model_path :
            ner_model_path = './_model/default'

        int_existence = data.check_exists(ner_model_path)
        if int_existence == -1:
            logger.error(f"Trained NER model does not exist at path {ner_model_path}")
            raise ValueError(f"File {ner_model_path} does not exist",
                              "Ending program")

        ner_model_abspath = os.path.abspath(ner_model_path)
        self.ner_pipeline = entity_recognizer.load_model(ner_model_abspath)

        # TODO: need to complete this part
        # Call entity_recognizer run ner model
        # Call select entities
        self.detected_ner = {}

        return 0

    def save_output(self,
                    save_format: str='json') -> None:
        
        dict_output = processing.format_output(dict_ners=self.detected_ner, list_org_lines=self.list_lines,
                                               dict_info=self.data_info)
        
        # Check if directory path exists
        int_existence = data.check_directory_path(path_directory=self.dir_output)
        if int_existence == 0:
            logger.info(f"Created output path {self.dir_output}")
        elif int_existence == -1:
            logger.warning(f"Directory {self.dir_output} does not exist\nSaving to path './output'")

        self.path_output = os.path.join(self.dir_output, f'{self.file_output}.json')

        data.save_file(output_data=dict_output, save_path=self.path_output)
        logger.info(f"Output saved to {self.path_output}")