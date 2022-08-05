from pipeline.constants import LOAD_METHODS

class Loader():
	def __init__(self):
		pass

	def load(self, data, method, **kwargs):
		if method not in LOAD_METHODS:
			raise Exception(f"In pipeline.loader.load(): Load method {method} is not supported")
		if method == "return":
			return data
		if method == "save":
			data.to_csv(kwargs["filepath"])