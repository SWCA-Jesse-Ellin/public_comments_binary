import argparse
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.models import load_model
import pandas as pd
import pathlib

from models.bilstm import BiLSTMModel

parser = argpare.ArgumentParser()
parser.add_argument("-f", "--input-file", type=pathlib.Path, required=True, help="Path to csv file that stores the input data")
parser.add_argument("-k", "--primary-key", type=str, required=False, default="comment_text", help="String key used in the csv to denote letter text column")
