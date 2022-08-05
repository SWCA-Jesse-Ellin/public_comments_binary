from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader

import spacy
import pandas as pd

class Pipeline():
	def __init__(self, primary_key="letter_text"):
		self.extractor = Extrtactor()
		self.transformer = Transformer(primary_key=primary_key)
		self.loader = Loader()

	def process(self, filepath, sep="sentence", method="return"):
		self.extractor.extract(filepath)
		data = self.transformer.transform(self.extractor.dump())
		return self.loader.load(data, method=method)