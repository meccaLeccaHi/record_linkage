# python NeuralNetwork class (modeled after examples in 'Make Your Own Neural Network'[2016]) 

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

# neural network class definition
class NeuralNetwork(linkage_tools.Linker):
		
	# initialise the neural network
	def __init__(self, features, hiddennodes, outputnodes, **keywords):
		
		# Set a classifier descriptors
		self.classifier_type = 'neural network'
		self.features = features
		if ('learningrate' in keywords):
			self.lr = keywords['learningrate'] # Learning rate
		else:
			self.lr = 0.1

		# set number of nodes in each input, hidden, output layer
		self.inodes = len(features)
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		# link weight matrices, wih and who
		# weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
		# w11 w21
		# w12 w22 etc 
		self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
		self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

		# activation function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)

		self.train_precision_list = []
		self.train_recall_list = []
		self.train_fscore_list = []
		self.train_accuracy_list = []

		self.val_precision_list = []
		self.val_recall_list = []
		self.val_fscore_list = []
		self.val_accuracy_list = []

	# train the neural network
	def train(self, inputs_list, truth_list, guess_list):
		record_list = inputs_list*.98+.01 # scale inputs to between .01 and .99
		truth_list = truth_list*.98+.01
		for record,truth in zip(record_list.values.tolist(),truth_list):

			# convert inputs list to 2d array
			inputs = np.array(record, ndmin=2, dtype=np.float).T
			truth = np.array(truth, ndmin=2, dtype=np.float)

			# calculate signals into hidden layer
			hidden_inputs = np.dot(self.wih, inputs)
			# calculate the signals emerging from hidden layer
			hidden_outputs = self.activation_function(hidden_inputs)

			# calculate signals into final output layer
			final_inputs = np.dot(self.who, hidden_outputs)
			# calculate the signals emerging from final output layer
			final_outputs = self.activation_function(final_inputs)

			# output layer error is the (truth - guesses)
			output_errors = truth - final_outputs
			# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
			hidden_errors = np.dot(self.who.T, output_errors) 

			# update the weights for the links between the hidden and output layers
			self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

			# update the weights for the links between the input and hidden layers
			self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	# query the neural network
	def query(self, inputs_list, input_ids):
		# convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2, dtype=np.float).T

		# calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# calculate the signals emerging from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# calculate signals into final output layer
		final_inputs = np.dot(self.who, hidden_outputs)

		# calculate the signals emerging from final output layer
 		final_outputs = self.activation_function(final_inputs)

		# Convert to linkage score to linkage 'cost'
		score_cost = abs(final_outputs - final_outputs.max())

		# Maximize pair-wise linkage scores (minimize cost with Munkres)
		link_list = input_ids.assign(linkage_cost = score_cost.T)
		winner_ind = self.maximize(link_list)

		# Return boolean array
		winners = np.zeros(len(link_list), np.bool)
		winners[winner_ind] = 1

		# Convert to binary output
		class_outputs = final_outputs>.5

		return winners.T, final_outputs.T # class_outputs.T
