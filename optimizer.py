import theano.tensor as T
from utils import sharedX
import theano
import numpy as np
class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay=0.005, momentum=0.9):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class AdagradOptimizer(object):
    def __init__(self, learning_rate, eps=1e-8):
        self.lr = learning_rate
        self.eps = eps

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
	for p, g in zip(params, grads):
	    value = p.get_value(borrow=True)
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),broadcastable = p.broadcastable)
            accu_new = accu + g ** 2
            updates.append((accu,accu_new))
	    updates.append((p,p-(self.lr * g/T.sqrt(accu_new + self.eps))))
            #updates[p] = p - (learning_rate * g /T.sqrt(accu_new + self.epsilon))
        return updates 
