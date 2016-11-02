import theano.tensor as T
import numpy as np

class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        # Your codes here
	
	return -(np.multiply(np.log(inputs),labels)).sum()
	# hint: labels are already in one-hot form
