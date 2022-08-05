import pandas as pd

class Extractor():
	def __init__(self):
		self.data = None

	def dump(self):
		return self.data

	def extract(self, filepath):
		self.data = pd.read_csv(filepath, encoding_errors="ignore")