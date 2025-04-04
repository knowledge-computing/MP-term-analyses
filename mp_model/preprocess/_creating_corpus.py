from Typing import Dict

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus, ClassificationCorpus

def csv_corpus(column_mapping:Dict[int, str],
               bool_skip_header:bool=True, delimiter:str=',',
               path_data_folder:str=None):
    """

    Parameters
    : column_mapping: 
    : bool_skip_header:
    : delimiter:
    : path_data_folder: 

    TODO: add support for when file name is not train,dev,test or combined into one with the split indicator
    """
    
    if path_data_folder:
        corpus: Corpus = CSVClassificationCorpus(data_folder=path_data_folder,
                                                 column_name_map=column_mapping,
                                                 skip_header=bool_skip_header,
                                                 delimiter=delimiter)
    return corpus

def ft_corpus(column_mapping:Dict[int, str],
              bool_skip_header:bool=True, delimiter:str=',',
              path_data_folder:str=None):
    return 0

# Load custom dataset
# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data'

# column format indicating which columns hold the text and label(s)
column_name_map = {4: "text", 1: "label_topic", 2: "label_subtopic"}


# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(data_folder,
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt',
                                      label_type='topic',
                                      )


# this is the folder in which train, test and dev files reside
data_folder = '/path/to/data/folder'

# load corpus by pointing to folder. Train, dev and test gets identified automatically.
corpus: Corpus = ClassificationCorpus(data_folder,
                                      label_type='topic',
                                      )

