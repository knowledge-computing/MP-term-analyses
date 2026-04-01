import os
import json

def check_directory_path(path_directory:str, bool_create=True) -> int:
    """
    Checks whether use defined (or default) directory exists in system
    
    Parameters
    : path_directory (str): path to directory
    : bool_create (bool): (optional) create directory if not exist, default True

    Return
    : (int): 0 if directory is created, 1 if directory exists, -1 if directory does not exist
    """
    if not os.path.exists(path_directory):
        if bool_create:
            os.makedirs(path_directory)
            return 0
        else:
            os.makedirs('./outputs')
            return -1

    return 1

def save_file(output_data:dict,
              path_save:str) -> int:
    """
    Saves file to user defined path
    
    Parameters
    : output_data (dict): zooniverse output data
    : path_save (str): path to save data on local

    Return
    : (int): 1 if file saved, -1 if failed
    """
    try:
        with open(path_save, 'w') as f:
            f.write(json.dumps(output_data))

        return 1
    
    except:
        return -1
    
def save_boto():
    return 0