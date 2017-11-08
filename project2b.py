from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import math

corruption_level=0.1
training_epochs = 5 #25
learning_rate = 0.6 #0.1
batch_size = 128
momentum=0.1
penalty=0.5
sparsity=0.05
with_attributes = True # use momentum, and sparsity?

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


def init_velocity(filter_shape):
    w_values =  np.zeros(shape=filter_shape, dtype=theano.config.floatX)
    return theano.shared(w_values, borrow=True)


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


def create_velocity(config):
    velocities = [0]
    for i in range(1, len(config)):
        v = init_velocity((config[i-1], config[i]))
        velocities.append(v)
    return velocities

def create_encoder_trainer(y,w,b,bp,layerIndex, inputX):
    global learning_rate

    # to train hidden encoder layer i, use output of layer (i-1)
    # for training

    z1 = T.nnet.sigmoid(T.dot(y[layerIndex], w[layerIndex].transpose()) + bp[layerIndex])
    x = y[layerIndex-1]
    cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) # general cost
    params1 = [w[layerIndex], b[layerIndex], bp[layerIndex]]
    grads1 = T.grad(cost1, params1)
    updates1 = [(param1, param1 - learning_rate * grad1)
            for param1, grad1 in zip(params1, grads1)]
    train_da1 = theano.function(inputs=[inputX], outputs = cost1, updates = updates1, 
        allow_input_downcast = True)
    return train_da1

def create_encoder_trainer_sparsity(y,w,b,bp,v,layerIndex,inputX):
    global learning_rate
    global momentum
    global sparsity
    global penalty

    beta = penalty
    rho = sparsity

    # to train hidden encoder layer i, use output of layer (i-1)
    # for training

    z1 = T.nnet.sigmoid(T.dot(y[layerIndex], w[layerIndex].transpose()) + bp[layerIndex])
    x = y[layerIndex-1]

    # calculate sparsity
    dh = beta*T.shape(z1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
			- beta*rho*T.sum(T.log(T.mean(z1, axis=0)+1e-6)) \
			- beta*(1-rho)*T.sum(T.log(1-T.mean(z1, axis=0)+1e-6))

    cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) + dh
    params1 = [w[layerIndex], b[layerIndex], bp[layerIndex]]

    gradW = T.grad(cost=cost1, wrt=w[layerIndex])
    gradB = T.grad(cost=cost1, wrt=[b[layerIndex],bp[layerIndex]])

    updates1 = []
    temp = momentum*v[layerIndex] - gradW*learning_rate

    updates1.append( (v[layerIndex], temp) )
    updates1.append( (w[layerIndex], w[layerIndex]+temp) )
    updates1.append( (b[layerIndex], b[layerIndex] - learning_rate * gradB[0]) )
    updates1.append( (bp[layerIndex], bp[layerIndex] - learning_rate * gradB[1]) )

    train_da1 = theano.function(inputs=[inputX], outputs = [cost1], updates = updates1, 
        allow_input_downcast = True)
    return train_da1


# load data
trX, teX, trY, teY = mnist(12000,2000)

x = T.dmatrix('x')  
d = T.dmatrix('d')

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

# create weights and biases
config = [28*28, 900, 625, 400, 10]

#config = [28*28, 900, 10]
weight_image_size = [28, 30, 25, 20]

weights,biases,bprimes = create_network_parameters(config)

if(with_attributes):
    velocities = create_velocity(config)

encoder_layer_count = len(config)-2

tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                dtype=theano.config.floatX)*x

# create the feedforward expressions
y=[tilde_x]
for i in range(1,len(config)):
    if i<len(config)-1:
        y.append(T.nnet.sigmoid(T.dot(y[i-1], weights[i]) + biases[i])) # hidden layers
    else:
        y.append(T.nnet.softmax(T.dot(y[i-1], weights[i]) + biases[i])) # softmax output layer

# create training functions for each encoder layer
train_enc=[]
for i in range(encoder_layer_count):
    if with_attributes:
        func = create_encoder_trainer_sparsity(y,weights,biases,bprimes,velocities,i+1, x)
    else:
        func = create_encoder_trainer(y,weights,biases,bprimes, i+1, x)
    train_enc.append(func)

# create encoder/decoder network for reconstruction of image
t0 = y[encoder_layer_count]
recoutputs = []
for i in range(encoder_layer_count):
    wi = encoder_layer_count-i
    t1 = T.nnet.sigmoid(T.dot(t0, weights[wi].transpose()) + bprimes[wi])
    t0 = t1
    recoutputs.append(y[i+1])
reconstruct_func = theano.function(inputs=[x], outputs = [t0]+recoutputs, allow_input_downcast = True)

# create training functions for feed forward network
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
            cs = train_enc[i](trX[start:end])
            c.append(cs)
          
            
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
    sz = weight_image_size[i]
    for j in range(100):
        pylab.subplot(10, 10, j+1); pylab.axis('off'); pylab.imshow(w1[:,j].reshape(sz,sz))
    pylab.savefig(filename_prefix +"enc_weights_%d.png" % (i+1))


# test images reconstruction
cmap = pylab.cm.get_cmap('gray')
figInputImages, axesInput = pylab.subplots(10,10)
figInputImages.suptitle("input images")

figOutputImages, axesOutput = pylab.subplots(10,10)
figOutputImages.suptitle("reconstructed images")

figHiddenActivation = []
axesActivation = []

for i in range(encoder_layer_count):
    f,ax = pylab.subplots(10,10)
    f.suptitle("hidden layer activation (%d neurons)" % config[i+1])

    figHiddenActivation.append(f)
    axesActivation.append(ax)

for i in range(100):
    ind = np.random.randint(low=0, high=2000) # select a random image
    indX, indY = i%10, i//10

    # plot the input images
    image = teX[ind:ind+1, :]
    axesInput[indX, indY].imshow(image.reshape(28,28),cmap=cmap)
    axesInput[indX, indY].axis('off')

    # reconstruct images
    oimages = reconstruct_func(image)
    axesOutput[indX, indY].imshow(oimages[0].reshape(28,28),cmap=cmap)
    axesOutput[indX, indY].axis('off')

    # plot hidden activations
    for j in range(encoder_layer_count):
        ax = axesActivation[j][indX, indY]
        ax.axis('off')
        ax.imshow(oimages[j+1].reshape(weight_image_size[j+1],weight_image_size[j+1]))

figInputImages.savefig(filename_prefix +"input_images.png")
figInputImages.savefig(filename_prefix +"reconstructed_images.png")
for i in range(encoder_layer_count):
    figHiddenActivation[i].savefig(filename_prefix +"hidden_activation_%d.png" % config[i+1])

# train the ffn
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
