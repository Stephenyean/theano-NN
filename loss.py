import theano.tensor as T
import numpy as np

class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        # Your codes here
	
	#return -(np.multiply(np.log(inputs + 1e-10),labels)).sum()/np.size(inputs,0)
	return T.sum(T.nnet.categorical_crossentropy(inputs,labels))	
