import numpy as np
import scipy.io

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

import theano
import lasagne

import time

# Load data
data = scipy.io.loadmat('digits.mat')
X = data["X"]
y = data["y"]
y[y==10]=0

# Scale the data
#X = MinMaxScaler().fit_transform(X.astype(float))

# Split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)

def build_nn(input_var=None): # note we can name all the layers the same thing, since we only return the last one.
    layer_in = lasagne.Layers.InputLayer(shape=(1,400),input_var=input_var)
    layer_in_drop = lasagne.Layers.DropoutLayer(layer_in, p=0.2) # 20% dropout
    layer_hidden_1 = lasagne.Layers.DenseLayer(layer_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform()) # Initialize weights with Glorot's scheme (which is the default anyway)
    layer_hidden_1_dropout = lasagne.Layers.DropoutLayer(layer_hidden_1, p=0.5) # 50% dropout
    layer_hidden_2 = lasagne.Layers.DenseLayer(layer_hidden_1_dropout, num_units=800, nonlinearity=lasagne.nonlinearitues.rectify) # presumably there is no initialization here, or it's the default?
    layer_hidden_2_dropout = lasagne.Layers.DropoutLayer(layer_hidden_2, p=0.5) # 50% dropout again
    layer_out = lasage.Layers.DenseLayer(layer_hidden_2_dropout, num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return layer_out

def minibatches(inputs,outputs,batchsize,shuffle=False):
    assert len(inputs) == len(outputs)
    if shuffle:
        idx = np.arange(len(inputs))
        np.random.shuffle(idx)
    for start in range(0,len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = idx[start:start+batchsize]
        else:
            excerpt = slice(start,start+batchsize)
        yield inputs[excerpt], outputs[excerpt]


def main(epochs=500):

    # Prepare theano variables
    input_var = theano.tensor.matrix('inputs')
    output_var = theano.tensor.ivector('targets')

    # Build the netowrk
    network = build_nn(input_var)

    # Loss function, used for training purposes:
    pred = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction,output_var) # todo What about a sigmoid function?
    loss = loss.mean()
    # todo See lasagne.regularization

    # Weight update for training: Stochastic Gradient Descent (SGD) with Nesterov momentum
    # Note that lasagne offers many more todo Look up other weight update types
    thetas = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, thetas, learning_rate=0.01, momentum=0.9)

    # Loss expression for validation / testing
    # Note that we only do forward prop, leaving out dropout layers
    test_pred = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_pred, output_var)
    test_loss = test_loss.mean()
    test_acc = theano.tensor.mean(theano.tensor.eq(theano.tensor.argmax(test_pred, axis=1),output_var),dtype=theano.config.floatX)

    # Compile theano function performing a training step
    train_fn = theano.function([input_var,output_var], loss, updates=updates)

    # Compile theano function computing validation loss and accuracy
    val_fn = theano.function([input_var, output_var],[test_loss,_test_acc])

    # Now we begin:
    for epoch in range(epochs):
        train_error = 0
        train_batched = 0
        start_time = time.time()
        for batch in minibatches(Xtrain,ytrain,500,shuffle=True):
            inputs, outputs = batch
            train_err += train_fn(inputs, outputs)
            train_batches += 1

        # todo continue from line 299 in tutorial



