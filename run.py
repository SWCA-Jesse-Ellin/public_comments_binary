import argparse
import pathlib
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--input-file", type=pathlib.Path, required=True, help="Path to csv file that stores the input data")
parser.add_argument("-k", "--primary-key", type=str, required=False, default="letter_text", help="String key used in the csv to denote letter text column (defaults to \"letter_text\"")
parser.add_argument("-s", "--separation-method", type=str, required=False, default="window", help="Method uesd to separate letter text into processing chunks (defaults to \"window\")")
parser.add_argument("--binary-search", action="store_true")
parser.add_argument("--brute-force", action="store_true")
parser.add_argument("--compress-windows", action="store_true")
parser.add_argument("--reconstruct-text", action="store_true")
parser.add_argument("--output-file", type=str, required=False, default="binary_model_results.csv", help="Path to csv file that results will be stored in")
args = vars(parser.parse_args())
output_name = args["output_file"]

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd

from models.bilstm import BiLSTMModel
from models.constants import TOKENIZER_TIMESTAMP, MODEL_TIMESTAMP, SEQUENCE_LEN
from pipeline.constants import WINDOW_STEP, TEXT_REPLACEMENT, WINDOW_SIZE
from pipeline.pipe import Pipeline
from utils import collapse_windows, reconstruct_text

if args["binary_search"]:
	from collections import Counter
	from tqdm import tqdm

	performance = dict()
	left = 1
	right = SEQUENCE_LEN
	with open(f"models/saved_models/tokenizer_{TOKENIZER_TIMESTAMP}.json") as f:
		tokenizer = tokenizer_from_json(f.read())
	pipeline = Pipeline(args["primary_key"])
	word_index = tokenizer.word_index
	word_index = {k : (v+3) for k,v in word_index.items()}
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2
	word_index["<UNUSED>"] = 3
	model = BiLSTMModel(load_file=f"models/saved_models/weights/LSTM_custom_{MODEL_TIMESTAMP}")
	print(f"Running on sentence separation")
	data = pipeline.process(args["input_file"], sep="sentence")
	x = data[args["primary_key"]]
	x_tokens = tokenizer.texts_to_sequences(x)
	x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=SEQUENCE_LEN)
	results = model.predict(x_pad)
	binary = model.toBinary(results, threshold=0.5)
	data["significance"] = binary
	data["confidence"] = results
	best_ratio = len(data[data["significance"] == True]) / len(data["significance"])
	print(f"True ratio: {best_ratio:.2%}")
	best_window_ratio = best_ratio
	baseline = best_ratio

	while left <= right:
		window_size = left + (right - left) // 2
		performance[window_size] = Counter()
		best_ratio = baseline
		inner_right = window_size
		inner_left = 1
		while inner_left <= inner_right:
			step_size = inner_left + (inner_right-inner_left) // 2
			print(f"Running with step size ({inner_left}, {step_size}, {inner_right}) and window size ({left}, {window_size}, {right})")
			data = pipeline.process(args["input_file"], sep=args["separation_method"], step_size=step_size, window_size=window_size)

			x = data[args["primary_key"]]
			x_tokens = tokenizer.texts_to_sequences(x)
			x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=SEQUENCE_LEN)

			results = model.predict(x_pad)
			binary = model.toBinary(results, threshold=0.5)

			data["significance"] = binary
			data["confidence"] = results

			ratio = len(data[data["significance"]==True]["significance"]) / len(data["significance"])
			print(f"True ratio: {ratio:.2%}")
			if ratio > best_ratio:
				data.to_csv(f"binary_model_results_step_{step_size}_window_{window_size}.csv")
				inner_left = step_size + 1
				best_ratio = ratio
				performance[window_size][step_size] = ratio
				performance[window_size][-1] = ratio
			else:
				inner_right = step_size - 1

		print(f"Window size {window_size} had best ratio {performance[window_size][-1]:.2%}")
		if performance[window_size][-1] > best_window_ratio:
			best_window_ratio = performance[window_size][-1]
			right = window_size - 1
		else:
			left = window_size + 1

	performance_df = pd.DataFrame(columns=["window_size", "step_size", "ratio"])
	sizes = []
	steps = []
	ratios = []
	for w in tqdm(performance.keys()):
		for s in tqdm(performance[w].keys(),leave=False):
			sizes.append(w)
			steps.append(s)
			ratios.append(performance[w][s])
	performance_df["window_size"] = sizes
	performance_df["step_size"] = steps
	performance_df["ratio"] = ratios
	performance_df.to_csv("search_results.csv")
	exit()

elif args["brute_force"]:
	from collections import Counter
	from tqdm import tqdm

	performance = dict()
	best_window_ratio = 0
	with open(f"models/saved_models/tokenizer_{TOKENIZER_TIMESTAMP}.json") as f:
		tokenizer = tokenizer_from_json(f.read())
	pipeline = Pipeline(args["primary_key"])
	word_index = tokenizer.word_index
	word_index = {k : (v+3) for k,v in word_index.items()}
	word_index["<PAD>"] = 0
	word_index["<START>"] = 1
	word_index["<UNK>"] = 2
	word_index["<UNUSED>"] = 3
	model = BiLSTMModel(load_file=f"models/saved_models/weights/LSTM_custom_{MODEL_TIMESTAMP}")

	for window_size in range(1, SEQUENCE_LEN+1):
		performance[window_size] = Counter()
		best_ratio = 0
		raw_data = pipeline.process(args["input_file"], sep=args["separation_method"], step_size=1)
		
		x = raw_data[args["primary_key"]]
		x_tokens = tokenizer.texts_to_sequences(x)
		x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=SEQUENCE_LEN)
		for step_size in range(1, SEQUENCE_LEN+1):
			print(f"Running with step size {step_size} and window size {window_size}")			

			results = model.predict(x_pad[::step_size])
			binary = model.toBinary(results, threshold=0.5)
			data = raw_data.copy()

			data["significance"].iloc[::step_size] = binary
			data["confidence"].iloc[::step_size] = results

			ratio = len(data[data["significance"]==True]["significance"]) / len(data["significance"])
			print(f"True ratio: {ratio:.2%}")
			if ratio > best_ratio:
				data.to_csv(f"binary_model_results_step_{step_size}_window_{window_size}.csv")
				inner_left = step_size + 1
				best_ratio = ratio
				performance[window_size][step_size] = ratio
				performance[window_size][-1] = ratio
			else:
				inner_right = step_size - 1

		print(f"Window size {window_size} had best ratio {performance[window_size][-1]:.2%}")
		if performance[window_size][-1] > best_window_ratio:
			best_window_ratio = performance[window_size][-1]
			right = window_size - 1
		else:
			left = step_size + 1

	performance_df = pd.DataFrame(columns=["window_size", "step_size", "ratio"])
	sizes = []
	steps = []
	ratios = []
	for w in tqdm(performance.keys()):
		for s in tqdm(performance[w].keys(),leave=False):
			sizes.append(w)
			steps.append(s)
			ratios.append(performance[w][s])
	performance_df["window_size"] = sizes
	performance_df["step_size"] = steps
	performance_df["ratio"] = ratios
	performance_df.to_csv("search_results.csv")

	exit()

pipeline = Pipeline(args["primary_key"])
data = pipeline.process(args["input_file"], sep=args["separation_method"])

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

data.to_csv(output_name)

if args["compress_windows"]:
	text_df = pipeline.transformer.transform(pipeline.extractor.dump(), method="none")
	num_to_text = {text_df["letter_num"].iloc[i] : text_df[args["primary_key"]].iloc[i] for i in range(len(text_df))}
	data = collapse_windows(data, num_to_text, numbercol="letter_num", startcol="frame_start", endcol="frame_end", labelcol="significance", textcol=args["primary_key"], step_size=WINDOW_STEP)
	output_name = output_name.split('.')[0] + "_collapsed.csv"
	data.to_csv(output_name)

if args["reconstruct_text"]:
	orig_data = pipeline.extractor.dump()
	def replace(text):
		for k,v in TEXT_REPLACEMENT.items():
			text = text.replace(k,v)
		return text
	orig_data[args["primary_key"]] = orig_data[args["primary_key"]].transform(replace)
	output_name = output_name.split('.')[0] + "_reconstructed.csv"
	data = reconstruct_text(data, orig_data, startcol="frame_start", endcol="frame_end", textcol=args["primary_key"], numbercol="letter_num", window_size=WINDOW_SIZE)
	data.to_csv(output_name)