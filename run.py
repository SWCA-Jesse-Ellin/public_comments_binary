import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input-file", type=pathlib.Path, required=True, help="Path to csv file that stores the input data")
parser.add_argument("-k", "--primary-key", type=str, required=False, default="letter_text", help="String key used in the csv to denote letter text column (defaults to \"letter_text\"")
args = vars(parser.parse_args())

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd

from models.bilstm import BiLSTMModel
from models.constants import TOKENIZER_TIMESTAMP, MODEL_TIMESTAMP, SEQUENCE_LEN
from pipeline.pipe import Pipeline

pipeline = Pipeline(args["primary_key"])
data = pipeline.process(args["input_file"])

with open(f"models/saved_models/tokenizer_{TOKENIZER_TIMESTAMP}.json") as f:
	tokenizer = tokenizer_from_json(f.read())

x = data[args["primary_key"]]
word_index = tokenizer.word_index
word_index = {k : (v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
x_tokens = tokenizer.texts_to_sequences(x)
x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=SEQUENCE_LEN)

model = BiLSTMModel(load_file=f"models/saved_models/weights/LSTM_custom_{MODEL_TIMESTAMP}")

results = model.predict(x_pad)
binary = model.toBinary(results, threshold=0.5)

data["significance"] = binary
data["confidence"] = results

data.to_csv("binary_model_results.csv")