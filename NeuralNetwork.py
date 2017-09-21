# python NeuralNetwork class

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

# neural network class definition
class NeuralNetwork(linkage_tools.Linker):
    """
    Multi-layered neural network.
    Parameters
    ------------
    inputnodes : int
        Size of input layer (i.e. number of 'neurons')
    hiddennodes : int
        Size of hidden layer
    outputnodes : int
        Size of output layer
    learningrate : float
        Learning rate (between 0.0 and 1.0)

    Attributes
    ------------
    who : 1d-array
        Weights for the links between the hidden and output layers.
    wih : 1d-array
        Weights for the links between the input and hidden layers.

    activation_function : function
        Function to apply to inputs.
    """
    
    # Initialize the neural network
    def __init__(self, features, hiddennodes, outputnodes, **keywords):

        self.features = features
        if ('learningrate' in keywords):
            self.lr = keywords['learningrate'] # Learning rate
        else:
            self.lr = 0.1

        # Set number of nodes in each input, hidden, output layer
        self.inodes = len(features)
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Set link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # Activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # Initialize performance lists
        self = linkage_tools.Linker.init_lists(self)

    # Train the neural network
    def train(self, inputs_list, truth_list, guess_list):
        record_list = inputs_list*.98+.01 # scale inputs to between .01 and .99
        truth_list = truth_list*.98+.01
        for record,truth in zip(record_list.values.tolist(),truth_list):

            # Convert inputs list to 2d array
            inputs = np.array(record, ndmin=2, dtype=np.float).T
            truth = np.array(truth, ndmin=2, dtype=np.float)

            # Calculate signals into hidden layer
            hidden_inputs = np.dot(self.wih, inputs)
            # Calculate the signals emerging from hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)

            # Calculate signals into final output layer
            final_inputs = np.dot(self.who, hidden_outputs)
            # Calculate the signals emerging from final output layer
            final_outputs = self.activation_function(final_inputs)

            # Output layer error is the (truth - guesses)
            output_errors = truth - final_outputs
            # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
            hidden_errors = np.dot(self.who.T, output_errors) 

            # Update the weights for the links between the hidden and output layers
            self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

            # Update the weights for the links between the input and hidden layers
            self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # Query the neural network
    def query(self, inputs_list, input_ids):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2, dtype=np.float).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Convert to linkage score to linkage 'cost'
        score_cost = abs(final_outputs - final_outputs.max())

        # Maximize pair-wise linkage scores (minimize cost with Munkres)
        link_list = input_ids.assign(linkage_cost = score_cost.T)
        winner_ind = self.maximize(link_list)

        # Return boolean array
        winners = np.zeros(len(link_list), np.bool)
        winners[winner_ind] = 1

        return winners.T, final_outputs.T