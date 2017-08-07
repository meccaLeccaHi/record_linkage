# python Probabilistic class 

import numpy as np
#import scipy.special # for the sigmoid function expit()

class Probabilistic(Linker):
		
	# Initialize the probabilistic linker
	def __init__(self, inputnodes):

		# Set number of nodes in input layer
		self.inodes = inputnodes

		# Initialize m- and u-probabilites to random values
		self.m_probs = np.repeat(.9, inputnodes)
		self.u_probs = np.repeat(.1, inputnodes)

		# Activation function is the logarithmic function
		self.activation_function = lambda x: np.sum(np.log(x))
		
	# Train the probabilistic linker
	def train(self, inputs_list, targets_list, input_ids):
		'''e.g. pr.train(bool_table[feature_vals],bool_table['real_match'],bool_table[['newb_id','bc_id']])'''

		# Convert inputs list to 2d array
		inputs = np.array(inputs, ndmin=2).T
		targets = np.array(targets, ndmin=2).T

		# Calculate probabilities going into linker
		mprob_mat = np.tile(self.mprobs, inputs.shape)
		mprob_mat[inputs==0] = 1 - mprob_mat[inputs==0]
		uprob_mat = np.tile(self.uprobs, inputs.shape)
		uprob_mat[inputs==0] = 1 - uprob_mat[inputs==0]

		# Calculate the probabilistic linker predictions
		linkage_score = np.apply_along_axis(self.activation_function, 0, foo_probs/bar_probs)
		# Convert to linkage score to linkage 'cost'
		linkage_cost = abs(linkage_score - linkage_score.max())

		# Maximize pair-wise linkage scores (minimize cost with Munkres)
		link_list = input_ids.assign(linkage_cost = linkage_cost)
        winner_index = self.maximize(link_list)           

		mask = np.ones(len(winner_index), np.bool)
		mask[winner_index] = 0
		loser_index = np.where(mask)

		# Count number of matches for each field
		u_match = sum(inputs[loser_index,:]) 
		m_match = sum(inputs[winner_index,:]) 

		# Count total number of pairs
		u_pair = sum(inputs==False)
		m_pair = sum(inputs==True)

		# Update m- and u-probabilities
		self.u_probs = np.divide(u_match_count,u_pair_count)
		self.m_probs = np.divide(m_match_count,m_pair_count)

	# Query the probabilistic linker
	def query(self, inputs_list):
		'''e.g. pr.query(bool_table[feature_vals])'''

		# Convert inputs list to 2d array
		inputs = np.array(inputs, ndmin=2).T

		# Calculate probabilities going into linker
		mprob_mat = np.tile(self.mprobs, inputs.shape)
		mprob_mat[inputs==0] = 1 - mprob_mat[inputs==0]
		uprob_mat = np.tile(self.uprobs, inputs.shape)
		uprob_mat[inputs==0] = 1 - uprob_mat[inputs==0]

		# Calculate the probabilistic linker predictions
		linkage_score = np.apply_along_axis(self.activation_function, 0, foo_probs/bar_probs)
		# Convert to linkage score to linkage 'cost'
		linkage_cost = abs(linkage_score - linkage_score.max())

		# Maximize pair-wise linkage scores (minimize cost with Munkres)
		link_list = input_ids.assign(linkage_cost = linkage_cost)
        winner_index = self.maximize(link_list)           

		winners = np.zeros(len(winner_index), np.bool)
		winners[winner_index] = 1

		return winners

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
