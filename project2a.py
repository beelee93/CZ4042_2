from load import mnist
import numpy as np
import pylab 

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 2 convolution layer, 2 max pooling layer, 1 hidden and a softmax layer
 
np.random.seed(10)

learning_rate = 0.05
batch_size = 128
decay_param = 1e-4
noIters = 100
momentum = 0.1
part_number = 2
debug = False

def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_velocity(filter_shape, d_type):
    w_values =  np.zeros(shape=filter_shape, dtype=d_type)
    return theano.shared(w_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def create_model(X, lstWeights, lstBiases):
    pool_dim=(2,2)
    y=[] # outputs of layers
    # 0 - conv layer
    y.append(T.nnet.relu(
        conv2d(X,lstWeights[0]) + \
        lstBiases[0].dimshuffle('x',0,'x','x') 
    ))

    # 1 - max pool
    y.append( pool.pool_2d(y[0], pool_dim, ignore_border=True))

    # 2 - conv layer
    y.append(T.nnet.relu(
        conv2d(y[1],lstWeights[1]) + \
        lstBiases[1].dimshuffle('x',0,'x','x') 
    ))

    # 3 - max pool
    y.append( pool.pool_2d(y[2], pool_dim, ignore_border=True))

    # flatten the output into a Sample-by-Outputs matrix
    yf = T.flatten(y[3], outdim=2)

    # 4 - neuron hidden layer
    y.append(T.nnet.relu(
        T.dot(yf, lstWeights[2]) + lstBiases[2]
    ))

    # 5 - Softmax
    y.append(T.nnet.softmax(
        T.dot(y[4], lstWeights[3]) + lstBiases[3]
    ))

    return y

def sgd(cost, weights, biases, lr=0.05, decay=1e-4):
    params = weights+biases
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, weights, biases, velocities, 
    lr=0.05, decay=1e-4, momentum=0.1):
    gradW = T.grad(cost=cost, wrt=weights)
    gradB = T.grad(cost=cost, wrt=biases)
    updates = []

    for i in range(len(gradW)):
        updates.append(
            (velocities[i], 
            momentum*velocities[i]-(gradW[i] + decay*weights[i]) * lr)
        )
        updates.append(
            (weights[i], weights[i]+velocities[i])
        )
        updates.append(
            (biases[i],biases[i]-(gradB[i]+decay*biases[i])*lr)
        )
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels
    
def create_weights_biases(dtype,config):
    weights = []
    biases = []

    for con in config:
        if(len(con)==2):
            w,b = init_weights_bias2(con, dtype)
        else:
            w,b = init_weights_bias4(con, dtype)
        weights.append(w); biases.append(b)
    return weights,biases

def create_velocity(dtype,config):
    velocities = []
    for con in config:
        v = init_velocity(con, dtype)
        velocities.append(v)
    return velocities

# configure the network
if debug:
    config = (
        (3, 1, 9, 9),
        (3, 3, 5, 5),
        (27, 10),
        (10, 10)
    )
else:
    config = (
        (15, 1, 9, 9),  # 15 9x9 filters
        (20, 15, 5, 5), # 20 5x5 filters
        (180, 100),     # 100 neurons layer
        (100, 10)       # 10 neurons softmax
    )

# load data
trX, teX, trY, teY = mnist(12000,2000,onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

# define input and output variables
X = T.tensor4('X')
Y = T.matrix('Y')

# create weights and biases
weights, biases = create_weights_biases(X.dtype, config)
outputs = create_model(X, weights, biases)

if part_number==2:
    velocities = create_velocity(X.dtype, config)

py_x = outputs[5]
y_x = T.argmax(py_x, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))

if part_number==1:
    updates = sgd(cost, weights, biases, lr=learning_rate, decay=decay_param)
elif part_number==2:
    updates = sgd_momentum(cost, weights, biases, velocities, lr=learning_rate, decay=decay_param, 
        momentum=momentum)
else:
    raise IndexError("Part number out of bounds")

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], 
    outputs=[outputs[0], outputs[1], outputs[2], outputs[3]], 
    allow_input_downcast=True)

# training
a = []
c = []
for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)

    cumCost, iterTotal = 0, 0
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
        iterTotal += 1
        cumCost += cost
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))

    cumCost /= iterTotal
    c.append(cumCost)
    print('Epoch %d: Avg Cost=%.2e Accu=%3f' % (i+1,c[i],a[i]))


# accuracy figure
fig = pylab.figure()
pylab.plot(range(noIters), a, color='blue')

ax0 = fig.axes[0]
ax0.set_xlabel('epochs')
ax0.set_ylabel('test accuracy', color='blue')

ax1 = ax0.twinx()
ax1.plot(range(noIters), c, color='red')
ax1.set_ylabel('training cost', color='red')

pylab.savefig('fig_2a/part%d_accu.png' % part_number)

# feature maps plot
for plot_i in range(2):
    ind = np.random.randint(low=0, high=2000)
    conv0, pool0, conv1, pool1 = test(teX[ind:ind+1,:])

    # input image
    fig = pylab.figure()
    pylab.gray()
    pylab.axis('off')
    pylab.imshow(teX[ind,:].reshape(28,28))
    fig.suptitle('input image')
    pylab.savefig('fig_2a/part%d_input%d.png' % (part_number, plot_i))

    # feature maps
    if debug:
        pylab.figure()
        pylab.gray()
        for i in range(3):
            pylab.subplot(3, 1, i+1); pylab.axis('off'); pylab.imshow(conv0[0,i,:].reshape(20,20))
        pylab.suptitle('convolved feature maps')
        pylab.savefig('fig_2a/part%d_input%d_conv0.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(3):
            pylab.subplot(3, 1, i+1); pylab.axis('off'); pylab.imshow(pool0[0,i,:].reshape(10,10))
        pylab.suptitle('pooled feature maps')
        pylab.savefig('fig_2a/part%d_input%d_pool0.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(3):
            pylab.subplot(3, 1, i+1); pylab.axis('off'); pylab.imshow(conv1[0,i,:].reshape(6,6))
        pylab.suptitle('convolved feature maps')
        pylab.savefig('fig_2a/part%d_input%d_conv1.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(3):
            pylab.subplot(3, 1, i+1); pylab.axis('off'); pylab.imshow(pool1[0,i,:].reshape(3,3))
        pylab.suptitle('pooled feature maps')
        pylab.savefig('fig_2a/part%d_input%d_pool1.png' % (part_number, plot_i))
    else:
        pylab.figure()
        pylab.gray()
        for i in range(15):
            pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(conv0[0,i,:].reshape(20,20))
        pylab.suptitle('convolved feature maps')
        pylab.savefig('fig_2a/part%d_input%d_conv0.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(15):
            pylab.subplot(5, 3, i+1); pylab.axis('off'); pylab.imshow(pool0[0,i,:].reshape(10,10))
        pylab.suptitle('pooled feature maps')
        pylab.savefig('fig_2a/part%d_input%d_pool0.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(20):
            pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(conv1[0,i,:].reshape(6,6))
        pylab.suptitle('convolved feature maps')
        pylab.savefig('fig_2a/part%d_input%d_conv1.png' % (part_number, plot_i))

        pylab.figure()
        pylab.gray()
        for i in range(20):
            pylab.subplot(5, 4, i+1); pylab.axis('off'); pylab.imshow(pool1[0,i,:].reshape(3,3))
        pylab.suptitle('pooled feature maps')
        pylab.savefig('fig_2a/part%d_input%d_pool1.png' % (part_number, plot_i))
  
