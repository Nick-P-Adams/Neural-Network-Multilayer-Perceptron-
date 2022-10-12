# Nick Adams 00883496

## delete the `pass` statement in every function below and add in your own code. 
import numpy as np

# Various math functions, including a collection of activation functions used in NN.
class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        x = np.array(x)
        return np.tanh(x)

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        x = np.array(x)
        return (1 - np.square(MyMath.tanh(x)))

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        x = np.array(x)
        _v_sigmoid = np.vectorize(lambda s: 1 / (1 + np.exp(-s)))
        return _v_sigmoid(x)
    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        x = np.array(x)
        return np.multiply(MyMath.logis(x), (1 - MyMath.logis(x)))

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        x = np.array(x)
        return x

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        x = np.array(x)
        return np.ones(x.shape)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        x = np.array(x)
        _v_relu = np.vectorize(lambda s: max(0,s))
        return _v_relu(x)

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        if(x > 0):
            return 1
        else:
            return 0

    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        x = np.array(x)
        _v_relu_de_scaler = np.vectorize(MyMath._relu_de_scaler)
        return _v_relu_de_scaler(x)

    