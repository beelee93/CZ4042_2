import numpy as np
import pylab 

fig = pylab.figure()

pylab.xlabel('epochs')
pylab.ylabel('test accuracy')

colors = ['red','green','blue']

for i in range(3):
    a=np.load("accu_part%d.npy" % (i+1))
    pylab.plot(range(len(a)), a, color=colors[i])

pylab.show()