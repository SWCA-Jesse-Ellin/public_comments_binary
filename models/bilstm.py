import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

from models.constants import VOCAB_SIZE

class BiLSTMModel():
	def __init__(self, load_file=None):
		if load_file:
			self.model = tf.keras.models.load_model(load_file)
		else:
			self.generateModel()
		self.compile()

	def generateModel(self):
		pass

	def compile(self):
		self.model.compile(optimizer="adam", loss=self.customLoss(), metrics=["accuracy", tf.keras.metrics.Recall(name="recall")], run_eagerly=True)

	def customLoss(self):
		def f1(y_true, y_pred):
			y_pred = K.round(y_pred)
			tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
			tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
			fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
			fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

			p = tp / (tp + fp + K.epsilon())
			r = tp / (tp + fn + K.epsilon())

			f1 = 2*p*r / (p+r+K.epsilon())
			f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
			return K.mean(f1)

		def f1_loss(y_true, y_pred):
			y_true = tf.where(y_true, 1., 0.)
			tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
			tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
			fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
			fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

			p = tp / (tp + fp + K.epsilon())
			r = tp / (tp + fn + K.epsilon())

			f1 = 2*p*r / (p+r+K.epsilon())
			f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
			return 1 - K.mean(f1)

		return f1_loss

	def predict(self, x):
		return self.model.predict(x, batch_size=512)

	def toBinary(self, Y, threshold=0.5):
		return np.array([True if y[0] >= threshold else False for y in Y])