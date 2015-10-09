import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import matplotlib
import scipy.optimize
from sklearn.metrics import accuracy_score

np.random.seed(0) # Set NP random seed to 0 such that data is consistent for testing purposes
X, y = sklearn.datasets.make_moons(200, noise = 0.2) # Make dataset

# Set some initial variables
m = np.float(len(X)) # training set size
Input = X.shape[1] # Input layer dimensionality, aka the number of features describing each data point
Output = 2 # Output layer dimensionality - here the system is binary so we want two output layers. We could also use one, I suppose.
Lambda = 0.001 # Regularization parameter, anything >0.26942 causes linear decision boundary and < 0.0003 causes fail max iterations
Hidden = 3 # Number of hidden layers
y_matrix = np.eye(Output)[y] # Convert the 1s and 0s from the dataset into a matrix with two columns: one for 1s and one for 0s. This is redundant in this example, but will make it easier to scale this script for data sets with more output dimenions

# Write simple sigmoid function as the activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1 * z))

# Helper function to actually predict the output based on the final theta values
def predict(theta,x):
    # Unroll the theta values as before
    theta1 = theta[:(Input+1)*Hidden].reshape((Hidden,(Input+1)))
    theta2 = theta[(Input+1)*Hidden:].reshape((Output,(Hidden+1)))
    m = np.float(len(x))
    #Forward propagation (also as before!)
    a1 = np.concatenate((np.ones((1,m)).T,x),axis=1)
    z2 = a1.dot(theta1.T)
    a2 = np.concatenate((np.ones((1,m)).T,sigmoid(z2)),axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    # argmax returns the index of the largest value, ie the index of the largest probability
    # Because of the layout of y_matrix, the index is 1 or 0, which is the output value!
    # axis = 1 specifies to return the argmax of each row
    return np.argmax(a3, axis=1)

# Helper function to calculate the gradient and subsequently update the theta values
# Backpropagation finds the relative contribution of each theta to the total error in the predicted values
# This is regularized and multiplied by the learning rate alpha
# Finally, this is subtracted from the original thetas
# The last step is to re-roll the individual thetas into a 1D array to pass back to fmin_cg
def nn(theta):

    # Unroll thetas
    theta1 = theta[:(Input+1)*Hidden].reshape((Hidden,(Input+1))) # [Hiddden X Input + 1]
    theta2 = theta[(Input+1)*Hidden:].reshape((Output,(Hidden+1))) # [Output X Hidden + 1]

    # Forward propagation calculates the probabilities given the current values for theta
    # Matrix dimensions are indicated after each line - this helps keep track of the math
    # Note that a1 is the input layer with an additional bias unit (made by np.ones()), thus it has dimensions [m X Input + 1]
    a1 = np.concatenate((np.ones((1,m)).T,X),axis=1) # [m X Input + 1] = [200X3]
    z2 = a1.dot(theta1.T) # [m X Input + 1] = [200X3]
    a2 = np.concatenate((np.ones((1,m)).T,sigmoid(z2)),axis=1) # [m X Hidden + 1] = [200X4] (the bias unit is also added here, hence the + 1!)
    z3 = a2.dot(theta2.T) # [m X Output] = [200X2]
    a3 = sigmoid(z3) # [m X Output] = [200X2]

    # Cost function
    # Note that the regularization term is added, excluding theta values in column 1 - these are the bias uit thetas and they are not regularized.
    #if Lambda==0:
    #    J = np.sum(-y_matrix*np.log(a3) - (1-y_matrix)*(np.log(1-a3)))/m
    #else:
    J = np.sum(-y_matrix*np.log(a3) - (1-y_matrix)*(np.log(1-a3)))/m + ((Lambda/(2*m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)))


    # Backpropagation
    d3 = (a3 - y_matrix) # [m X Output] = [200X2]
    #d2 = (a3 - y_matrix).dot(theta2[:,1:]) * (a2[:,1:]*(1-a2[:,1:])) # [m X Hidden] = [200X3]
    d2 = (a3 - y_matrix).dot(theta2) * (a2*(1-a2)) # [m X Hidden] = [200X3]
    D1 = d2[:,1:].T.dot(a1)/m # [Hidden X Input + 1] = [3X3]
    D2 = d3.T.dot(a2)/m # [Input X Hidden + 1] = [2X4]
    if Lambda!=0:
        D1[:,1:] += (Lambda/m * theta1[:,1:]) # [Hidden X Input + 1] = [3X3]
        D2[:,1:] += (Lambda/m * theta2[:,1:]) # [Output X Hidden + 1] = [2X4]

    # Roll grads D1 and D2 into 1D array
    #theta = np.concatenate((D1.reshape(((Input+1)*Hidden)),D2.reshape(((Hidden+1)*Output))))
    grads = np.concatenate((D1.reshape(((Input+1)*Hidden)),D2.reshape(((Hidden+1)*Output))))

    return J, grads

def cost(theta):
    return nn(theta)[0]

def grad(theta):
    return nn(theta)[1]

# Function to randomly initialize theta values at the start of the whole process
# L_in and L_out are the input and output dimensions for that particular layer
# Thus L_in and L_out are not necessarily the final Input and Output
# i.e. theta1 bridges the Input layer (L_in in this case) and the hidden layer (which is L_out in this case)
# and theta2 bridges the hidden layer (L_in now) and the output layer (L_out now...)
def randtheta(L_in,L_out):
        np.random.seed(0)
        # The following is based on lecture material in Andrew Ng's Coursera course
        rand_epsilon = np.sqrt(6) / np.sqrt(L_in+L_out)
        theta = (np.random.random((L_out, L_in + 1)) *(2*rand_epsilon)) - rand_epsilon
        return theta

# Make 1D array of theta values to pass to fmin_cg
theta_init = np.concatenate((randtheta(Input,Hidden).reshape(((Input+1)*Hidden)),randtheta(Hidden,Output).reshape(((Hidden+1)*Output))))

# This function simply plots the results so that we can visualize the decision boundary and the original data
# It takes as it's only argument the predict function above
def plot(pred_func):

    # Set the min and max values, with some padding
    # This is based on the original data
    # Thus the boundaries of the plot do not extend beyond the actual data
    # We're going to feed dummy data into predict() to plot a decision boundary and we'll need to constrain that data
    # IMPORTANT: y_min / yy are not the y-values of the data set, they're the second feature which is plotted on the y-axis
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    h = 0.01 # this is the plotting interval of the dummy data. Smaller values make smoother curves, but the result is essentially the same.

    # Generate a grid of points with distance h between them
    # Specifically, meshgrid makes symmetrical matrices along the range supplied
    # Thus here, we get a symmetrical matrix of values along the x and y ranges above
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # This is equivalent of saying: predict(model,np.c_[xx.ravel(), yy.ravel()])
    # ravel returns a 1D array, np.c_ concatenates these arrays along their 2nd axis
    # The result is an array with two columns (1 for each feature, i.e. n columns)
    # There are many rows in this array: the (range from x_min:x_max)/h * (range from y_min:y_max)/h!
    # The lambda function that calls this whole function (inside plot_decision_boundary()) provides the theta argument for predict
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) # This outputs the predicted value for *each* row in the array
    Z = Z.reshape(xx.shape) # The predicted values are reshaped to [(range from x_min:x_max)/h X (range from y_min:y_max)/h]

    # Now we plot a contour plot with dimensions set using meshgrid, with the array of outcomes in Z making the contour line
    # The result is a colored plot showing where in the meshgrid the algorithm predicts 1 or 0
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Spectral) # The actual plots are simply overlayed so we can see how well it fits the data

# Now that the helper functions are all done, we run fmin_cg
# This is where most of my confusion currently lies
# fmin_cg takes as it's first argument the function that returns the value to be minimized
# The second argument is the 1D array of values to be optimized (specifically, their initial values)
# An optional third argument can specify the gradient function of f, but this doesn't work for me
# In it's current form, the function runs and returns a good decision boundary, however the grad() function is never called and thus Backpropagation never occurs. How is theta being updated? I know fmin_cg is doing that, but what is the math??

opt = scipy.optimize.fmin_cg(cost, x0=theta_init, fprime=grad, full_output=1, disp=1);
theta = opt[0]
print "The cost is %f" %(opt[1])
print "The accuracy is %f" %(accuracy_score(y, predict(theta,X))) # this comes from sklearn.metrics

# Lastly, this plots the decision boundary for easy understanding
plot(lambda x: predict(theta,x))
plt.title("Decision Boundary for hidden layer size 3")
