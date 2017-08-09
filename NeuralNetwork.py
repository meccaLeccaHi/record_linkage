# python NeuralNetwork class (modeled after examples in 'Make Your Own Neural Network'[2016]) 

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

# neural network class definition
class NeuralNetwork(linkage_tools.Linker):
		
	# initialise the neural network
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# set number of nodes in each input, hidden, output layer
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		# w11 w21
		# w12 w22 etc 
		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

		# learning rate
		self.lr = learningrate

		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

	# train the neural network
	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		# output layer error is the (target - actual)
		output_errors = targets - final_outputs
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors) 

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	# query the neural network
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# calculate the signals emerging from final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

'''
# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate is 0.3
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
In [5]:
# test query (doesn't mean anything useful yet)
n.query([1.0, 0.5, -1.5])
'''