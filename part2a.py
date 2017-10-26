#
# Dependencies
#
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


######################################################################

#
# Helper functions
#

def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]
     
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)


######################################################################

#
# CNN Class
#
class ConvNetwork:
    #
    # Constructor
    #
    def __init__(self, dLearningRate=0.5, dDecay=1e-6):
        # allocate variables
        self.trainFunc = None
        self.testFunc  = None
        self.learningRate = dLearningRate
        self.decayParam = dDecay

        self.W = [0] # weights
        self.B = [0] # biases
        self.Y = [ ] # layer outputs

        self.Input = T.tensor4('X')
        self.Output = T.matrix('Y')

        pass

    #
    # Construct the Network
    #
    def constructNetwork(self):
        # set up weights and biases
        self.createWeightsAndBiases()

        # output of layer 0 = input layer
        self.Y.append(self.Input)

        # create the expressions
        self.createExpressions()

        pass

    #
    # Populate the weights and biases list
    #
    def createWeightsAndBiases(self):
        # --- Input (28 x 28 image)

        # Layer 1 - Conv Layer
        # 15 20x20 feature maps, 9x9 filter
        w,b = init_weights_bias4( (15, 1, 9, 9), self.Input.dtype )
        self.W.append(w)
        self.B.append(b)

        # -- Max Pool 2x2 --
         # - 15 10x10 feature maps
        self.W.append(0) # this is appended for indexing convenience
        self.B.append(0)

        # Layer 2 - Conv Layer
        # 20 6x6 feature maps, 5x5 filter
        # (amounts to 300 feature maps)
        w,b = init_weights_bias4( (20, 1, 5, 5),self.Input.dtype )
        self.W.append(w)
        self.B.append(b)

        # -- Max Pool 2x2 --
        # - 300 3x3 feature maps
        self.W.append(0)
        self.B.append(0)

        # Layer 3 - Fully connected layer
        # 15*20*3*3 = 2700 neurons to 100 neurons
        w,b = init_weights_bias2( (15*20*3*3, 100),self.Input.dtype )
        self.W.append(w)
        self.B.append(b)

        # Layer 4 - Softmax output layer
        w,b = init_weights_bias2( (100, 10),self.Input.dtype )
        self.W.append(w)
        self.B.append(b)

        print("ConvNet.createWeightsAndBiases succeeded!")
        
    #
    # Creates the theano expressions
    #
    def createExpressions(self):
        # TODO: Populate

    #
    # Train the network
    #
    def train(self, trainInput, testInput):
        if self.trainFunc is None:
            raise Exception('trainFunc has not been set')

        pass


######################################################################

#
# CNN Class with Momentum
#
class ConvNetworkMomentum(ConvNetwork):
    #
    # Constructor
    #
    def __init__(self, dLearningRate=0.5, dDecay=1e-6, dMomentum=0.5):
        # call parent constructor
        ConvNetwork.__init__(self, dLearningRate=dLearningRate, dDecay=dDecay)

        # set instance variables
        self.momentum = 0.5
    

    #
    # Construct the Network
    #
    def constructNetwork(self):
        
        pass



######################################################################

#
# CNN Class with RMSProp
#
class ConvNetworkRMSProp(ConvNetwork):
    #
    # Constructor
    #
    def __init__(self,dLearningRate=0.001, dDecay=1e-4,dRoe=0.9, dEps=1e-6):
        # call parent constructor
        ConvNetwork.__init__(self,dLearningRate=dLearningRate, dDecay=dDecay)

        # set instance variables
        self.roe = dRoe
        self.epsilon = dEps
    

    #
    # Construct the Network
    #
    def constructNetwork(self):
        
        pass

######################################################################

#
# Main Entry Point
#
def main():
    # set a seed so results are reproducible
    np.random.seed(10)

    a = ConvNetwork()
  





######################################################################
if __name__ == '__main__':
    main()