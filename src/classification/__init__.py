from .classifiers import *
from .losses import *


def load_classifier():
	return LinearClassifier, LabelLoss

