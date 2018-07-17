#Logistic Regression with a Neural Network mindset
#Welcome to your first (required) programming assignment! You will build a logistic regression classifier to recognize cats. This assignment will step you through how to do this with a Neural Network mindset, and so will also hone your intuitions about deep learning.

#Instructions:

#Do not use loops (for/while) in your code, unless the instructions explicitly ask you to do so.
#You will learn to:

#Build the general architecture of a learning algorithm, including:
#Initializing parameters
#Calculating the cost function and its gradient
#Using an optimization algorithm (gradient descent)
#Gather all three functions above into a main model function, in the right order.

#Problem Statement: You are given a dataset ("data.h5") containing:

#- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#- a test set of m_test images labeled as cat or non-cat
#- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
#You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
##from lr_utils import load_dataset

### Loading the data (cat/non-cat)
##train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

### Example of a picture
##index = 25
##plt.imshow(train_set_x_orig[index])
##print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

##m_train = train_set_x_orig.shape[0]
##m_test = test_set_x_orig.shape[0]
##num_px = train_set_x_orig.shape[1]

##print ("Number of training examples: m_train = " + str(m_train))
##print ("Number of testing examples: m_test = " + str(m_test))
##print ("Height/Width of each image: num_px = " + str(num_px))
##print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
##print ("train_set_x shape: " + str(train_set_x_orig.shape))
##print ("train_set_y shape: " + str(train_set_y.shape))
##print ("test_set_x shape: " + str(test_set_x_orig.shape))
##print ("test_set_y shape: " + str(test_set_y.shape))

### Reshape the training and test examples
##### START CODE HERE ###
##train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
##test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
##### END CODE HERE ###

##print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
##print ("train_set_y shape: " + str(train_set_y.shape))
##print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
##print ("test_set_y shape: " + str(test_set_y.shape))
##print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

###One common preprocessing step in machine learning is to center and 
###standardize your dataset, meaning that you substract the mean of 
###the whole numpy array from each example, and then divide each 
###example by the standard deviation of the whole numpy array. 
###But for picture datasets, it is simpler and more convenient and 
###works almost as well to just divide every row of the dataset by 
###255 (the maximum value of a pixel channel).
##train_set_x = train_set_x_flatten/255.
##test_set_x = test_set_x_flatten/255.

### GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1. / (1 + np.exp(-z))
    
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### 
    w = np.zeros((dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### 
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z) # compute activation
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y)*np.log(1-A))  # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### 
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A - Y)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#Test Cases for Propagate
##w1, b1, X1, Y1 = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
##grads, cost = propagate(w1, b1, X1, Y1)
##print ("dw = " + str(grads["dw"]))
##print ("db = " + str(grads["db"]))
##print ("cost = " + str(cost))

#The results should be
##dw = [[ 0.99845601]
      ##[ 2.39507239]]
##db = 0.00145557813678
##cost = 5.801545319394553


#You have initialized your parameters.
#You are also able to compute a cost function and its gradient.
#Now, you want to update the parameters using gradient descent.
#Exercise: Write down the optimization function. The goal is to learn  
#ww  and  bb  by minimizing the cost function  JJ .
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


#The previous function will output the learned w and b. 
#We are able to use w and b to predict the labels for a dataset X. 
#Implement the predict() function. 
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ###
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z) # compute activation
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### 
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#Test Cases for Predict
##w1 = np.array([[0.1124579],[0.23106775]])
##b1 = -0.3
##X1 = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
##print ("predictions = " + str(predict(w1, b1, X1)))

#The results should be
##predictions = [[ 1.  1.  0.]]



#You will now see how the overall model is structured by putting together 
#all the building blocks (functions implemented in the previous parts) together, 
#in the right order.
#Exercise: Implement the model function. Use the following notation:
#Y_prediction_test for your predictions on the test set
#Y_prediction_train for your predictions on the train set
#w, costs, grads for the outputs of optimize()

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros 
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    # Gradient descent 
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d




#Test cases for model
# Some actual quantitative tests for the model function
##myw = d["w"]
##myb = d["b"]
##print ("Shape w = " + str(myw.shape))
##print ("Shape b = " + str(myb.shape))
##print ("w = " + str(myw))
##print ("b = " + str(myb))
##print ("w(range) = ", str(myw[(range(100,104),0)]))
##print ("w(range) = ", str(myw[(range(1000,1004),0)]))
##print ("Costs = " + str(d["costs"]))
##myPredTest = d["Y_prediction_test"]
##print ("Shape Y_prediction_test = " + str(myPredTest.shape))
##print ("Y_prediction_test = " + str(myPredTest[(0,range(10,16))]))
##myPredTrain = d["Y_prediction_train"]
##print ("Shape Y_prediction_train = " + str(myPredTrain.shape))
##print ("Y_prediction_train = " + str(myPredTrain[(0,range(20,26))]))

#results should be
##Shape w = (12288, 1)
##Shape b = ()
##w = [[ 0.00961402]
     ##[-0.0264683 ]
 ##[-0.01226513]
 ##..., 
 ##[-0.01144453]
 ##[-0.02944783]
 ##[ 0.02378106]]
##b = -0.0159062439997
##w(range) =  [-0.02656382 -0.00940811 -0.01162105 -0.02210741]
##w(range) =  [-0.03129008 -0.02407051 -0.00264713 -0.02803747]
##Costs = [array(0.6931471805599453), array(0.5845083636993087), array(0.4669490409465546), array(0.37600686694802066), array(0.3314632893282512), array(0.3032730674743828), array(0.27987958658260487), array(0.2600421369258757), array(0.2429406846779662), array(0.22800422256726074), array(0.2148195137844964), array(0.20307819060644994), array(0.19254427716706862), array(0.1830333379688351), array(0.17439859438448874), array(0.16652139705400332), array(0.15930451829756614), array(0.1526673247129651), array(0.1465422350398234), array(0.1408720757031016)]
##Shape Y_prediction_test = (1, 50)
##Y_prediction_test = [False False  True  True False  True]
##Shape Y_prediction_train = (1, 209)
##Y_prediction_train = [False False False False  True  True]