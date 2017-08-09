# python Perceptron class

import numpy as np
import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

class Perceptron(linkage_tools.Linker):
	"""Perceptron classifer.
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

	# Train the neural network
	def train(self, inputs_list, targets_list):
		''' pt.train(bool_table[feature_vals],bool_table['real_match'])'''

		# Convert inputs list to 2d array and transpose
		inputs = np.array(inputs_list, ndmin=2)
		inputs = np.concatenate((inputs,np.repeat(1, len(inputs[:,1]))[:, None]), axis=1) # Add bias input

		targets = np.array(targets_list, ndmin=2)

		#final_outputs = self.query(inputs_list)		

		# Output error (answer - guess)
		output_errors = targets - self.query(inputs_list)

		# Update the weights
		for i,row in enumerate(inputs):
			self.weights += self.lr * output_errors[:,i] * row
			#print 'output_errors[:,i] : ' + str(output_errors[:,i])
			#print self.lr * output_errors[:,i] * row
		# self.weights += self.lr * np.dot(output_errors, inputs)
		
		# Update the sum-of-squares
		self.tss.append(np.sum(output_errors**2))

	# Query the neural network
	def query(self, inputs_list):
		''' pt.query(bool_table[feature_vals])'''

		# Convert inputs list to 2d array and transpose
		inputs = np.array(inputs_list, ndmin=2).T
		inputs = np.vstack([inputs,np.repeat(1, len(inputs[1,:]))]) # Add bias input

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
