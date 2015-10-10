import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import train_test_split
from scipy.optimize import fmin_cg
from sklearn.metrics import accuracy_score

# import matplotlib

# Load data Set
data = scipy.io.loadmat('ex3data1.mat')
X = data["X"]
y = data["y"]
y[y==10]=0

# Split th data into testing and training sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)

# Set some initial variables
m = np.float(X.shape[0]) # total number of examples
mtrain = np.float(Xtrain.shape[0]) # Number of training examples
mtest = np.float(Xtest.shape[0]) # Number of testing examples
n = np.float(X.shape[1]) # Number of features
Output = 10 # Output layer dimensionality - here the system is binary so we want two output layers. We could also use one, I suppose.
Lambda = 0.01 # Regularization parameter, anything >0.26942 causes linear decision boundary and < 0.0003 causes fail max iterations
Hidden = 10 # Number of hidden layers
ytrain_matrix = np.eye(Output)[ytrain].reshape(mtrain,Output)
ytest_matrix = np.eye(Output)[ytest].reshape(mtest,Output)

# Visualize the data
def drawplot(draw):
    if draw:
        idx = np.random.randint(m,size=100)
        fig, ax = plt.subplots(10, 10)
        img_size = math.sqrt(n)
        for i in range(10):
            for j in range(10):
                Xi = X[idx[i*10+j],:].reshape(img_size, img_size).T
                ax[i, j].set_axis_off() # Turns off the axes for all the subplots
                ax[i,j].imshow(Xi, aspect='auto',cmap='gray')
        plt.show()
drawplot(False)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1 * z))

def predict(theta,x):
    # Unroll the theta values as before
    theta1 = theta[:(n+1)*Hidden].reshape((Hidden,(n+1)))
    theta2 = theta[(n+1)*Hidden:].reshape((Output,(Hidden+1)))
    
    m = np.float(x.shape[0])
    
    #Forward propagation (also as before!)
    a1 = np.concatenate((np.ones((1,m)).T,x),axis=1)
    z2 = a1.dot(theta1.T)
    a2 = np.concatenate((np.ones((1,m)).T,sigmoid(z2)),axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    
    return np.argmax(a3, axis=1)

def nn(theta):

    # Unroll thetas
    theta1 = theta[:(n+1)*Hidden].reshape((Hidden,(n+1))) # [Hiddden X n + 1] = [10X401]
    theta2 = theta[(n+1)*Hidden:].reshape((Output,(Hidden+1))) # [Output X Hidden + 1] = [10X11]

    # Forward propagation
    a1 = np.concatenate((np.ones((1,mtrain)).T,Xtrain),axis=1) # [m X n + 1] = [3750X401]
    z2 = a1.dot(theta1.T) # [m X Hidden] = [3750X10]
    a2 = np.concatenate((np.ones((1,mtrain)).T,sigmoid(z2)),axis=1) # [m X Hidden + 1] = [3750X11]
    z3 = a2.dot(theta2.T) # [m X Output] = [3750X10]
    a3 = sigmoid(z3) # [m X Output] = [3750X10]

    # Cost function
    J = np.sum(-ytrain_matrix*np.log(a3) - (1-ytrain_matrix)*(np.log(1-a3)))/mtrain + ((Lambda/(2*mtrain)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2)))
        

    # Backpropagation
    d3 = (a3 - ytrain_matrix) # [m X Output] = [3750X10]
    d2 = (a3 - ytrain_matrix).dot(theta2) * (a2*(1-a2)) # [m X Hidden+1] = [3750X11]
    D1 = d2[:,1:].T.dot(a1)/mtrain # [Hidden X n + 1] = [10X401]
    D2 = d3.T.dot(a2)/mtrain # [n X Hidden + 1] = [10X11]
    if Lambda!=0:
        D1[:,1:] += (Lambda/mtrain * theta1[:,1:]) # [Hidden X n + 1] = [3X3]
        D2[:,1:] += (Lambda/mtrain * theta2[:,1:]) # [Output X Hidden + 1] = [2X4]

    # Roll grads D1 and D2 into 1D array
    #theta = np.concatenate((D1.reshape(((n+1)*Hidden)),D2.reshape(((Hidden+1)*Output))))
    grads = np.concatenate((D1.reshape(((n+1)*Hidden)),D2.reshape(((Hidden+1)*Output))))

    return J, grads

def cost(theta):
    return nn(theta)[0]

def grad(theta):
    return nn(theta)[1]

def randtheta(L_in,L_out):
        np.random.seed(0)
        rand_epsilon = np.sqrt(6) / np.sqrt(L_in+L_out)
        theta = (np.random.random((L_out, L_in + 1)) *(2*rand_epsilon)) - rand_epsilon
        return theta

theta_init = np.concatenate((randtheta(n,Hidden).reshape(((n+1)*Hidden)),randtheta(Hidden,Output).reshape(((Hidden+1)*Output))))

opt = fmin_cg(cost, x0=theta_init, fprime=grad, full_output=1, disp=1, maxiter=300);
theta = opt[0]
print "The cost is %f" %(opt[1])
print "The accuracy is on the training set is %f" %(accuracy_score(ytrain, predict(theta,Xtrain))) # this comes from sklearn.metrics
print "The accuracy is on the testing set is %f" %(accuracy_score(ytest, predict(theta,Xtest))) # this comes from sklearn.metrics

# Lastly, this plots the decision boundary for easy understanding
#plot(lambda x: predict(theta,x))
#plt.title("Decision Boundary for hidden layer size 3")
