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

from mp_model import data, classification, identification, preprocess
# from fusemine import training

from mp_model._utils import (
    DefaultLogger,
)
import mp_model._save_utils as save_utils

logger = DefaultLogger()
logger.configure("INFO")

class MpModel:
    def __init__(self) -> None:
        return 0