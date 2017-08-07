# python Perceptron class

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

class Perceptron(linkage_tools.Linker):

	# Initialize the perceptron
	def __init__(self, inputnodes, learningrate):

		self.inodes = inputnodes # Set number of nodes in input layer
		self.lr = learningrate # Learning rate
		self.tss = 0 # Initialize sum-of-squares

		# Initialize perceptron weights to random values (1..n+1 to accomodate bias neuron)
		self.weights = np.random.uniform(low=-0.01, high=0.01, size=inputnodes+1)

		# Activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

	# train the neural network
	def train(self, inputs_list, targets_list):
		''' pt.train(bool_table[feature_vals],bool_table['real_match'])'''

		# convert inputs list to 2d array
		inputs = np.array(inputs, ndmin=2).T
		targets = np.array(targets, ndmin=2).T

		''' 
		bool_table['links_neural'] = bool_table.apply(activation, axis=1, features=feature_vals)
		weights = [weights[key]+learning_rate*error*row[key] for key in feature_vals+['bias']]
		'''

		# Calculate signals into perceptron
		final_inputs = np.dot(self.weights, inputs)

		# Calculate the perceptron predictions
		final_outputs = self.activation_function(final_inputs)

		# Output error (target - actual)
		output_errors = targets - final_outputs

		# Update the weights
		self.weights += self.lr * np.dot(output_errors, np.transpose(hidden_outputs))		

		# Update the sum-of-squares
		self.tss += error * error

	# query the neural network
	def query(self, inputs_list):
		''' pt.query(bool_table[feature_vals])'''

		# Convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# Calculate signals into perceptron
		final_inputs = np.dot(self.weights, inputs)
		# Calculate the perceptron predictions
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

'''
# number of input nodes
input_nodes = 5

# learning rate is 0.3
learning_rate = 0.3

# create instance of perceptron
p = Perceptron(input_nodes, learning_rate)
# test query (doesn't mean anything useful yet)
p.query([1.0, 0.5, -1.5])
'''


"""
# DELETE ME (BELOW)
def activation(row,features):
    ''' Get prediction of perceptron for each row of cross-joined blocks (pandas dataframe) '''
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    activ = sigmoid(sum([row[key]*weights[key] for key in features+['bias']])) # sigmoid function
    
    return activ

def update_weights(row,features):
    ''' Update weights of perceptron for each row of cross-joined blocks (pandas dataframe) '''
    error = row['real_match'] - row['links_neural'] # target - output
    
    weights = [weights[key]+learning_rate*error*row[key] for key in feature_vals+['bias']]
    return weights

    #new_tss = tss + error * error # add squared error
"""
