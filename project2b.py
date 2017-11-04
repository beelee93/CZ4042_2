from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

corruption_level=0.1
training_epochs = 25
learning_rate = 0.1
batch_size = 128
with_attributes = False # use momentum, and sparsity?

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

def create_network_parameters(config):
    """ config=(input_count, enc1, ..., encN, output_count) """
    w=[0] # zero is added for indexing convenience
    b=[0]
    bp=[0]

    for i in range(1, len(config)):
        w.append(init_weights(config[i-1], config[i]))
        b.append(init_bias(config[i]))

    for i in range(len(config)-2):
        bp.append(init_bias(config[i]))

    return w,b,bp

def create_encoder_trainer(y,w,b,bp,layerIndex, inputX):
    global learning_rate

    # to train hidden encoder layer i, use output of layer (i-1)
    # for training

    z1 = T.nnet.sigmoid(T.dot(y[layerIndex], w[layerIndex].transpose()) + bp[layerIndex])
    x = y[layerIndex-1]
    cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))
    params1 = [w[layerIndex], b[layerIndex], bp[layerIndex]]
    grads1 = T.grad(cost1, params1)
    updates1 = [(param1, param1 - learning_rate * grad1)
            for param1, grad1 in zip(params1, grads1)]
    train_da1 = theano.function(inputs=[inputX], outputs = cost1, updates = updates1, 
        allow_input_downcast = True)
    return train_da1

# load data
trX, teX, trY, teY = mnist(12000,2000)

x = T.fmatrix('x')  
d = T.fmatrix('d')

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# create weights and biases
#config = [28*28, 900, 625, 400, 10]
config = [28*28,900,10]
weights,biases,bprimes = create_network_parameters(config)

encoder_layer_count = len(config)-2

# -- create the training functions --
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                dtype=theano.config.floatX)*x

# 1 - create the feedforward expressions
y=[tilde_x]
for i in range(1,len(config)):
    if i<len(config)-1:
        y.append(T.nnet.sigmoid(T.dot(y[i-1], weights[i]) + biases[i])) # hidden layers
    else:
        y.append(T.nnet.softmax(T.dot(y[i-1], weights[i]) + biases[i])) # softmax output layer

# 2 - create training functions for each encoder layer
train_enc=[]
for i in range(encoder_layer_count):
    train_enc.append(create_encoder_trainer(y,weights,biases,bprimes, i+1, x))

# 3 - create training functions for feed forward network
p_y2 = y[len(config)-1] # final layer output
y2 = T.argmax(p_y2, axis=1)
cost2 = T.mean(T.nnet.categorical_crossentropy(p_y2, d))

params2 = []
for i in range(1,len(config)):
    params2.append(weights[i])
    params2.append(biases[i])

grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]

train_ffn = theano.function(inputs=[x, d], outputs = cost2, updates = updates2, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = y2, allow_input_downcast=True)

filename_prefix = "fig_2b/"
if(with_attributes):
    filename_prefix += "wa_"


# train each encoder layer
for i in range(encoder_layer_count):
    print("** training dae%d ..." % (i+1) )
    d = []
    for epoch in range(training_epochs):
        # go through training set
        c = []
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            c.append(train_enc[i](trX[start:end]))
        d.append(np.mean(c, dtype='float64'))
        print(d[epoch])

    # training stats
    pylab.figure()
    pylab.plot(range(training_epochs), d)
    pylab.xlabel('iterations')
    pylab.ylabel('cross-entropy')
    pylab.savefig(filename_prefix +"enc_train_%d.png" % (i+1))

    # weights visualized
    w1 = weights[i+1].get_value()
    pylab.figure()
    pylab.gray()
    for j in range(100):
        pylab.subplot(10, 10, j+1); pylab.axis('off'); pylab.imshow(w1[:,j].reshape(28,28))
    pylab.savefig(filename_prefix +"enc_weights_%d.png" % (i+1))

"""
TODO:
    for i = 1 to 100:
        select a test image
        run it through encoder stack
        plot out the reconstructed image
        plot out the hidden layer activation matrix
"""


print('\n** training ffn ...')
d, a = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    d.append(np.mean(c, dtype='float64'))
    a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
    print(a[epoch])

fig = pylab.figure()
pylab.plot(range(training_epochs), d, color='red')
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy', color='red')

ax = fig.axes[0].twinx()
ax.plot(range(training_epochs), a, color='green')
ax.set_ylabel('test accuracy', color='green')
fig.savefig(filename_prefix +"ffn_training.png")

"""
w2 = W2.get_value()
pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(w2)
pylab.savefig('figure_2b_5.png')
"""