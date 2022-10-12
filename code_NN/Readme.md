# Implementation of the forwardfeed neural network using stochastic gradient descent via back-propagation.

You will do the coding in these files: `math_util.py` and `nn.py`. 


## One real-world data set in CSV format: `MNIST` -- the handwriting single digits. 

The provided notebook already provides the necessary data preparation work, unless you want to have your own notebook for testing.

Reminder and suggestion: 

- When you read in the data set into numpy array, do not misalign/mess up the samples and their labels. 

- When you randomly shuffle the each data set (in sub-project 2), do not misalign/mess up the samples and their labels.

- An easy way to read in CSV files is to use the `Pandas` library. 



## Sub-project 1 (30 points)

Finish the following function in the `MyMath` class: `tanh`, `logis`, `iden`, and `relu`. Test them to ensure they do what they are supposed to do.

Finish the following function in the `NeuralNetwork` class: `add_layer`, `_init_weights`.



## sub-project 2 (50 points)

Finish the following functions in the `NeuralNetwork` class: `predict`, and `error`.

The majority of the code for the `predict` and `error` functions is to implement a procedure called `forward feed`, which will also be used in the NN's training that you will implement later. So I suggest you cut out the `forward feed` code separately for reuse purpose.

After you finish sub-projects 1&2, if you comment out the training part in the notebook, your NN is ready to make prediction for new samples, which of course will be of no quality because we haven't trained the network yet. 

## sub-project 3.1 (20 points)

Finish the following function in the `MyMath` class: `tanh_de`, `logis_de`, `iden_de`, and `relu_de`. Test them to ensure they do what they are supposed to do.

## sub-project 3.2 (200 points)
Finish the `fit` function in the `nn.py` file. 


Note: if you use the provided notebook for training/testing, your model shall have a predication accuracy of at least 97% for the validation MNIST data set. 