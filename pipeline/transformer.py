from pipeline.constants import TEXT_REPLACEMENT, SEP_METHODS, WINDOW_STEP, WINDOW_SIZE
from models.constants import SEQUENCE_LEN

import re
import pandas as pd
import spacy
sp = spacy.load("en_core_web_sm")
from tqdm import tqdm

class Transformer():
	def __init__(self, primary_key="letter_text"):
		self.stopwords = sp.Defaults.stop_words
		self.primary_key = primary_key
		
	def transform(self, data, method="sentence", step_size=WINDOW_STEP, window_size=WINDOW_SIZE):
		self.step_size = step_size
		self.window_size = window_size
		if method not in SEP_METHODS:
			raise Exception(f"In pipeline.transformer.transform(): separation method {method} is not supported")
		data = self.removeBlanks(data)
		data = self.process(data, method=method)
		data = self.removeBlanks(data)
		return data

	def removeBlanks(self, data):
		filter = data[self.primary_key] != ""
		data = data[filter]
		return data.dropna()

	def process(self, data, method="sentence"):
		text = []
		item_numbers = []
		letter_numbers = []
		frame_start = []
		frame_end = []
		for i in tqdm(range(len(data))):
			letter = data[self.primary_key].iloc[i]
			new_text = self.splitText(letter, method=method)
			if method == "window":
				newer_text = new_text["text"]
				new_frames = new_text["frames"]
				frame_start += new_frames[0]
				frame_end += new_frames[1]
				new_text = newer_text
			text += new_text
			new_item_count = [i+1 for i in range(len(new_text))]
			item_numbers += new_item_count
			letter_numbers += [data["letter_num"].iloc[i]] * len(new_item_count)

		if method == "window":
			return pd.DataFrame({self.primary_key : text,
								 "letter_num" : letter_numbers,
								 "sequence_num" : item_numbers,
								 "frame_start" : frame_start,
								 "frame_end" : frame_end})
		return pd.DataFrame({self.primary_key : text,
							 "letter_num" : letter_numbers,
							 "sequence_num" : item_numbers})

	def splitText(self, text, method="sentence"):
		new_text = []
		if method == "sentence":
			doc_text = [entry.strip() for entry in re.split("[.!?]+", text)]
			for entry in tqdm(doc_text, leave=False):
				new_text.append(self.processText(entry))
		elif method == "window":
			step_size = self.step_size
			doc_text = self.processText(text)
			new_text = dict()
			new_text["text"] = [doc_text[i:i+self.window_size] for i in tqdm(range(0, len(doc_text), step_size), leave=False)]
			new_text["frames"] = [[i for i in range(0, len(doc_text), step_size)], [i+self.window_size for i in range(0, len(doc_text), step_size)]]
		elif method == "none":
			new_text = [self.processText(text)]
		return new_text

	def processText(self, text):
		for k,v in TEXT_REPLACEMENT.items():
			text = text.replace(k,v)
		text = re.sub("[^a-zA-Z0-9]", ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text = " ".join([word for word in text.split(' ') if word not in self.stopwords])
		return text.strip().lower()
