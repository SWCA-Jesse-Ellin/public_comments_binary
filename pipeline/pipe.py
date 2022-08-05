from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader

import spacy
import pandas as pd

class Pipeline():
	def __init__(self):
		self.extractor = Extrtactor()
		self.transformer = Transformer()
		self.loader = Loader()

	def process(self, filepath, sep="sentence", primary_key="letter_text"):
		self.extractor.extract(filepath)
		self.transformer.transform(self.extractor.dump(), primary_key=primary_key)
		return self.loader.load(self.transformer.dump(), method="return")