# python Perceptron class

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

class Perceptron(linkage_tools.Linker):
	"""
	Perceptron classifer.
    Parameters
    ------------
    inputnodes : int
        Size of perceptron (i.e. number of 'neurons')
	learningrate : float
        Learning rate (between 0.0 and 1.0)

    Attributes
    ------------
    weights : 1d-array
        Perceptron weights.
    activation_function : function
        Function to apply to perceptron inputs.
    """

	# Initialize the perceptron
	def __init__(self, inputnodes, learningrate):

		self.inodes = inputnodes # Set number of nodes in input layer
		self.lr = learningrate # Learning rate
		self.tss = [] # Initialize sum-of-squares

		# Initialize perceptron weights to random values (1..n+1 to accomodate bias neuron)
		self.weights = np.random.uniform(low=-0.01, high=0.01, size=inputnodes+1)

		# Activation function is the sigmoid function
		self.activation_function = lambda x: (x>0)*1.0

		self.prec_list = []
		self.recall_list = []
		self.iter_qual_list = [0.0]

	# Train the neural network
	def train(self, inputs_list, winner_index, input_ids):
		''' e.g. pt.train(bool_table[feature_vals],bool_table['real_match'],bool_table[['newb_id','bc_id']])'''

		# Convert inputs list to 2d array and transpose
		inputs = np.array(inputs_list, ndmin=2)
		inputs = np.concatenate((inputs,np.repeat(1, len(inputs[:,1]))[:, None]), axis=1) # Add bias input
	
		guesses, _ = self.query(inputs_list, input_ids)		

		# Output error (answer - guess)
		output_errors = np.subtract(winner_index, guesses)

		# Update the weights
		#for i,row in enumerate(inputs):
		#	self.weights += self.lr * output_errors[i] * row
		#	#print 'output_errors[:,i] : ' + str(output_errors[:,i])
		#	#print self.lr * output_errors[:,i] * row

		#self.weights += self.lr * np.dot(output_errors, inputs)
		self.weights = self.weights + self.lr * np.dot(output_errors, inputs)

		# signals = self.lr * output_errors[0] * row[0]
		# print 'guess/answers/signals : ' + str([guesses[0],winner_index[0],signals])

		# Update the sum-of-squares
		self.tss.append(np.sum(output_errors**2))

	# Query the neural network
	def query(self, inputs_list, input_ids):
		''' pt.query(bool_table[feature_vals],bool_table[['newb_id','bc_id']])'''

		# Convert inputs list to 2d array and transpose
		inputs = np.array(inputs_list, ndmin=2).T
		inputs = np.vstack([inputs,np.repeat(1, len(inputs[1,:]))]) # Add bias input

		# Calculate signals into perceptron
		perc_score = np.dot(self.weights, inputs)
		# Calculate the perceptron predictions
		final_outputs = self.activation_function(perc_score)

		# Convert to linkage score to linkage 'cost'
		perc_cost = abs(perc_score - perc_score.max())

		# Maximize pair-wise linkage scores (minimize cost with Munkres)
		link_list = input_ids.assign(linkage_cost = perc_cost)
		#print 'link_list : ' + str(link_list)	
		winner_index = self.maximize(link_list)
		
		# Return boolean array
		winners = np.zeros(len(link_list), np.bool)
		winners[winner_index] = 1

		#print 'activation function outputs : ' + str(np.where(final_outputs))
		#print 'munkres outputs : ' + str(winner_index)

		return winners, perc_score

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

