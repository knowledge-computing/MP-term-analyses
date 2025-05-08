import os
from typing import Union, List, Dict, Tuple

import boto3

def check_exist(path_file:str) -> int:
    """
    Checks if the given input path exists

    Parameter
    : path_file (str): path to the input file

    Return
    : (int): integer indicating if the input path exists in the os
    """
    if os.path.exists(path_file):
        return 1
    
    return -1


def load_local_data(path_file:str) -> List[str]:
    """
    Loading data from local

    Parameter
    : path_file (str): path to input_file

    Return
    : (List[str]): list of lines in the file
    """
    f = open(path_file, 'r')
    list_lines = f.readlines()
    f.close()

    return list_lines

def load_boto_data(boto_key:str, 
                   boto_bucket:str='covenants-deed-images') -> List[str]:
    """
    Loading data from S3 bucket based on boto_key

    Parameter
    : boto_key (str): Location from boto_bucket to the data (including file extension)
    : boto_bucket (str): S3 bucket name where the data is loacted, default: covenants-deed-images

    Return
    : (List[str]): list of lines in the file
    """
    
    s3 = boto3.client('s3')

    content_object = s3.get_object(Bucket=boto_bucket, Key=boto_key)
    list_lines = content_object['Body'].read().decode('utf-8').split('\n')

    return list(list_lines)