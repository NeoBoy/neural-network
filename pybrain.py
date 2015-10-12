# THIS IS WIP. DO NOT RUN.


from pybrain.tools.shortcuts import buildNetwork

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

from pybrain.supervised.trainers import BackpropTrainer


trainer = BackpropTrainer(net, ds)
trainer.train()
# This call trains the net for one full epoch and returns a double proportional to the error.

trainer.trainUntilConvergence()
# This returns a whole bunch of data, which is nothing but a tuple containing the errors for every training epoch.