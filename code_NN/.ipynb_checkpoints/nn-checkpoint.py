# Nick Adams 00883496

### Delete every `pass` statement below and add in your own code. 

# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 
import numpy as np
import math
from code_NN.nn_layer import NeuralLayer
import code_NN.math_util as mu
import code_NN.nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        newLayer = code_NN.nn_layer.NeuralLayer(d, act)
        self.layers.append(newLayer)
        self.L += 1
        
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        #weight_rng = np.random.default_rng(2142)
        
        for i in range(1, self.L+1):
            sqrtD = math.sqrt(self.layers[i-1].d)
            limit = 1/sqrtD
            self.layers[i].W = np.random.uniform(-limit, limit, (self.layers[i-1].d + 1, self.layers[i].d)) 
            
            
    def _ff(self, X):
        self.layers[0].X = np.insert(X, 0, 1, axis=1)
        
        for i in range(1, (self.L+1)):
            cur_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            cur_layer.S = prev_layer.X @ cur_layer.W
            cur_layer.X = cur_layer.act(cur_layer.S)
            cur_layer.X = np.insert(cur_layer.X, 0, 1, axis=1)
            
        return self.layers[self.L].X[:, 1:]
    
    def _back_prop(self, Y, N_prime):
        for i in range(self.L-1, 0, -1):
            self.layers[i].Delta = self.layers[i].act_de(self.layers[i].S) * \
            (self.layers[i+1].Delta @ (self.layers[i+1].W).T)[:,1:]
            self.layers[i].G = np.einsum('ij,ik -> jk', self.layers[i-1].X, self.layers[i].Delta) * \
            (1/N_prime)
    
    def _update_weights(self, eta):
        for i in range(1, (self.L+1)):
            self.layers[i].W = self.layers[i].W - (eta * self.layers[i].G)
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 
        
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        
        X_prime = X
        Y_prime = Y
        N_prime = mini_batch_size
        
        current_step = 0
        for t in range(1, iterations):
            if(SGD == True):
                # data set should be randomly shuffled first however it is not here for ease of testing
                # code for SGD logistic regression has code that I could move here to implement
                # random shuffling of data set
                current_step += mini_batch_size
                current_step, X_prime, Y_prime = self.getMiniBatch(X, Y, current_step, N_prime)
            
            X_output = self._ff(X_prime)
            self.layers[self.L].Delta = 2 * (X_output - Y_prime) * \
            self.layers[self.L].act_de(self.layers[self.L].S)
            
            self.layers[self.L].G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X, self.layers[self.L].Delta)\
            * (1/N_prime)
            
            self._back_prop(Y_prime, N_prime)
            self._update_weights(eta)
        
    def getMiniBatch(self, X, Y, current_step, N_prime):
        n,d = X.shape
        min_index = current_step - N_prime
        
        if(n - current_step < 0):
            # If we step over all the samples keep up to n-1 then wrap to top 
            #  and concatenate remaining mini_batch to bottom.
            n_max_index = N_prime - (n - min_index)
            X_prime = np.concatenate((X[min_index : n], X[0 : n_max_index]))
            Y_prime = np.concatenate((Y[min_index : n], Y[0 : n_max_index]))
            current_step = 0
        else:
            # Else we just need from min_index to current_step which will be mini_batch samples in size
            X_prime = X[min_index : current_step]
            Y_prime = Y[min_index : current_step]
        
        return current_step, X_prime, Y_prime
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        predictions = self._ff(X)
        predicted_Labels = np.argmax(predictions, axis=1)
        
        return predicted_Labels
        
        
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        sample_Labels = np.argmax(Y, axis = 1)
        predicted_Labels = self.predict(X)
        
        error = np.sum(sample_Labels != predicted_Labels)
        
        return error/len(sample_Labels)
     