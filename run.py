import argparse
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequence
from tensorflow.keras.models import load_model
import pandas as pd
import spacy

from models.bilstm import BiLSTMModel
