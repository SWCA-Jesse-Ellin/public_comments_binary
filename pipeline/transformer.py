from pipeline.constants import TEXT_REPLACEMENT, SEP_METHODS, SEQUENCE_LEN, WINDOW_STEP

import re
import pandas as pd
import spacy
sp = spacy.load("en_core_web_sm")
from tqdm import tqdm

class Transformer():
	def __init__(self, primary_key="letter_text"):
		self.stopwords = sp.Defaults.stop_words
		self.primary_key = primary_key

	def transform(self, data, method="sentence"):
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
		letter_count = 1
		for letter in tqdm(data[self.primary_key].tolist()):
			new_text = self.splitText(letter, method=method)
			text += new_text
			new_item_count = [i+1 for i in range(len(new_text))]
			item_numbers += new_item_count
			letter_numbers += [letter_count] * len(new_item_count)
			letter_count += 1

		return pd.DataFrame({self.primary_key : text,
							 "letter_number" : letter_numbers,
							 "sequence_number" : item_numbers})

	def splitText(self, text, method="sentence"):
		new_text = []
		if method == "sentence":
			doc_text = [entry.strip() for entry in re.split("[.!?]+", text)]
			for entry in tqdm(doc_text, leave=False):
				new_text.append(self.processText(entry))
		else if method == "window":
			window_size = SEQUENCE_LEN
			step_size = WINDOW_STEP
			doc_text = self.processText(text)
			new_text = [doc_text[i:i+window_size] for i in tqdm(range(0, len(doc_text), step_size), leave=False)]
		return new_text

	def processText(self, text):
		for k,v in TEXT_REPLACEMENT.items():
			text = text.replace(k,v)
		text = re.sub("[^a-zA-Z0-9]", ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text = " ".join([word for word in text.split(' ') if word not in self.stopwords])
		return text.strip().lower()
