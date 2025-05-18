import os
import warnings
from typing import Union, Dict

import polars as pl
from polars.exceptions import MapWithoutReturnDtypeWarning
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=MapWithoutReturnDtypeWarning)

from mpterm import data, processing, entity_recognizer, tars_classifier

from mpterm._utils import (
    DefaultLogger,
    str_to_json_extracted
)

logger = DefaultLogger()
logger.configure("INFO")

class MPTerm:
    def __init__(self,
                 input_info: Union[str, dict],
                 dir_output:str=None,
                 file_output:str=None,
                 verbose: bool=False) -> None:
        
        self.dir_output = dir_output
        self.file_output = file_output
        self.bool_ocr_correct = False

        # Need to setup for case where bool_local is true
        self.path_data, self.uuid = str_to_json_extracted(input_info)

        if verbose:
            logger.set_level('DEBUG')
        else:
            logger.set_level('WARNING')

    def load_data(self,
                  bool_local:bool=False) -> None:
        """
        Creating data information dictionary and loading the actual data

        Argument
        : bool_local (bool) - (optional) Using local data
        """
        updated_path, workflow, lookup = processing.get_components(self.path_data)
        self.data_info = {'workflow': workflow, 'lookup': lookup, 'uuid': self.uuid}

        if bool_local:
            int_existence = data.check_exist(self.path_data)
            if int_existence == -1:
                logger.error(f"File {self.path_data} does not exist")
                raise ValueError(f"File {self.path_data} does not exist",
                                "Ending program")

            self.list_lines = data.load_local_data(self.path_data)
        else:
            self.list_lines = data.load_boto_data(updated_path)
        logger.info(f"File {self.path_data} loaded\nTotal of {len(self.list_lines)} lines")

        # Converting list of lines to sentences
        self.list_sentences = processing.to_sentence(input_strs=self.list_lines)

        # Identifying beginning and end of line of each sentences
        self.dict_line_num = processing.get_line_num(list_sentences=self.list_sentences,
                                                     list_lines=self.list_lines)

    def entity_recog(self, 
                     ner_model_path:str=None) -> None:
        """
        Main model running entity recognition

        Argument
        : ner_model_path (str) - (optional) User defined ner model path
        """
        if not ner_model_path :
            ner_model_path = './mpterm/_model/default'

        int_existence = data.check_exist(ner_model_path)
        if int_existence == -1:
            logger.error(f"Trained NER model does not exist at path {ner_model_path}")
            raise ValueError(f"File {ner_model_path} does not exist",
                              "Ending program")

        ner_model_abspath = os.path.abspath(ner_model_path)
        self.ner_pipeline = entity_recognizer.load_model(ner_model_abspath)

        # Running entity recognizer model
        ner_result = entity_recognizer.run_nermodel(ner_pipeline=self.ner_pipeline,
                                                    input_sentence=self.list_sentences)
        
        # Clean NER entitites
        detected_ner_p_sentence = entity_recognizer.select_entities(ner_results=ner_result)

        # Format output to Zooniverse output format
        self.dict_output = processing.format_output(dict_ners=detected_ner_p_sentence, 
                                                    dict_line_num=self.dict_line_num, 
                                                    list_lines=self.list_lines,
                                                    dict_info=self.data_info)
        
    def doc_classify(self,
                     tars_model_path:str=None) -> None:
        """
        Main model running entity recognition

        Argument
        : tars_model_path (str) - (optional) User defined tars model path
        """
        if not tars_model_path :
            tars_model_path = './mpterm/_model/tars/best-model.pt'

        int_existence = data.check_exist(tars_model_path)
        if int_existence == -1:
            logger.error(f"Trained NER model does not exist at path {tars_model_path}")
            raise ValueError(f"File {tars_model_path} does not exist",
                              "Ending program")

        tars_model_abspath = os.path.abspath(tars_model_path)
        self.tars_pipeline = tars_classifier.load_model(tars_model_abspath)

        # Running document classification model
        tars_result = tars_classifier.run_tarsmodel(tars_pipeline=self.tars_pipeline,
                                                    input_sentence=self.list_sentences)
        
        # TODO: Verify if works
        

    def save_output(self,) -> None:
        """
        Saves zooniverse output to user defined path
        """
        # Check if directory path exists
        int_existence = data.check_directory_path(path_directory=self.dir_output)
        if int_existence == 0:
            logger.info(f"Created output path {self.dir_output}")
        elif int_existence == -1:
            logger.warning(f"Directory {self.dir_output} does not exist\nSaving to path './output'")

        self.path_output = os.path.join(self.dir_output, f'{self.file_output}.json')

        data.save_file(output_data=self.dict_output, path_save=self.path_output)
        logger.info(f"Output saved to {self.path_output}")

    def return_output(self,) -> dict:
        """
        Returns zooniverse output
        """

        return self.dict_output