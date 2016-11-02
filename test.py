from layers import Softmax
import numpy as np

m=np.array([(1,2),(3,5)])
s=Softmax('s')
print s.forward(m)
print m.sum(axis=1)
print m
