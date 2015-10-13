# THIS IS WIP. DO NOT RUN.
# Installed conda install --channel https://conda.binstar.org/derickl pybrain
# This gives me version 0.3.3

import scipy.io
import numpy as np
from PyBrain.datasets import ClassificationDataSet

import pybrain

# Load data
data = scipy.io.loadmat('digits.mat')
X = data["X"]
y = data["y"]
y[y==10]=0

# Make data-specific paramters
m = X.shape[0]
n = X.shape[1]
classes = 10
y_matrix = np.eye(classes)[y].reshape(m,classes)

# Model parameters
nodes=10

nndata = ClassificationDataSet(m,classes)
for i in range(m):
    nndata.addSample(X[i,:],y[i])

fnn = buildNetwork(n,nodes,classes,bias=True,outclass=SoftmaxLayer)
print fnn






from pybrain_nn.tools.shortcuts import buildNetwork
from pybrain_nn.structure import FeedForwardNetwork
n = FeedForwardNetwork()

from pybrain_nn.structure import LinearLayer, SigmoidLayer
inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

from pybrain_nn.structure import FullConnection
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

n = 10 # number of input nodes
nodes = 3 # number of hidden layer nodes
classes = 2 # number of output classes

# The build network shortcut
net = buildNetwork(n,nodes,classes)
# Default is Sigmoid squashing function
# Can use a tanh layer: hiddenclass=TanhLayer
# Can use a bias layer: bias=True

# The network is already intialized with random values, so we can calculate it's output
net.activate([2, 1]) # expects a list, tuple or an array as input.


# The layers are named automatically
net['in']
net['hidden0']
net['out']
net['bias']

from pybrain_nn.supervised.trainers import BackpropTrainer


trainer = BackpropTrainer(net, ds)
trainer.train()
# This call trains the net for one full epoch and returns a double proportional to the error.

trainer.trainUntilConvergence()
# This returns a whole bunch of data, which is nothing but a tuple containing the errors for every training epoch.