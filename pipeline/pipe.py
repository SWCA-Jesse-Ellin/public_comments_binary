from pipeline.extractor import Extractor
from pipeline.transformer import Transformer
from pipeline.loader import Loader
from pipeline.constants import WINDOW_STEP
from models.constants import SEQUENCE_LEN

import spacy
import pandas as pd

class Pipeline():
	def __init__(self, primary_key="letter_text"):
		self.extractor = Extractor()
		self.transformer = Transformer(primary_key=primary_key)
		self.loader = Loader()

	def process(self, filepath, sep="sentence", method="return", step_size=WINDOW_STEP, window_size=SEQUENCE_LEN):
		self.extractor.extract(filepath)
		data = self.transformer.transform(self.extractor.dump(), method=sep, step_size=step_size, window_size=window_size)
		return self.loader.load(data, method=method)