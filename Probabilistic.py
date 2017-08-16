# python Probabilistic class 

import numpy as np
#import scipy.special # for the sigmoid function expit()
import linkage_tools # for the Linker superclass

class Probabilistic(linkage_tools.Linker):
		
	# Initialize the probabilistic linker
	def __init__(self, inputnodes):

		# Set number of nodes in input layer
		self.inodes = inputnodes

		# Initialize m- and u-probabilites to random values
		self.m_probs = np.repeat(.9, inputnodes)
		self.u_probs = np.repeat(.1, inputnodes)

		# Activation function is the logarithmic function
		self.activation_function = lambda x: np.sum(np.log(x))

		self.precision_list = []
		self.recall_list = []
		self.iter_qual_list = [0.0]
		
	# Train the probabilistic linker
	def train(self, inputs_list, winner_index):
		'''e.g. pr.train(bool_table[feature_vals],bool_table['pair_match'])'''

		# Convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2)
		
		loser_index = np.ones(len(inputs_list), np.bool)
		loser_index[np.where(winner_index)] = 0
		
		# Count number of matches for each field
		u_match = [sum(i) for i in zip(*inputs[np.where(loser_index)])]
		m_match = [sum(i) for i in zip(*inputs[np.where(winner_index)])]

		# Count total number of pairs
		u_pair = np.sum(loser_index)
		m_pair = np.sum(winner_index)

		# Update m- and u-probabilities
		self.m_probs = np.true_divide(m_match,m_pair)
		self.u_probs = np.true_divide(u_match,u_pair)

		# print self.m_probs,self.u_probs

	# Query the probabilistic linker
	def query(self, inputs_list, input_ids):
		'''e.g. pr.query(bool_table[feature_vals],bool_table[['newb_id','bc_id']])'''

		# Convert inputs list to 2d array
		inputs = np.array(inputs_list, ndmin=2)

		# Calculate probabilities going into linker
		mprob_mat = np.tile(self.m_probs, (inputs.shape[0],1))
		mprob_mat[inputs==0] = 1 - mprob_mat[inputs==0]
		uprob_mat = np.tile(self.u_probs, (inputs.shape[0],1))
		uprob_mat[inputs==0] = 1 - uprob_mat[inputs==0]

		# Calculate the probabilistic linker predictions
		linkage_score = np.apply_along_axis(self.activation_function, 1, mprob_mat/uprob_mat)
		# Convert to linkage score to linkage 'cost'
		linkage_cost = abs(linkage_score - linkage_score.max())

		# Maximize pair-wise linkage scores (minimize cost with Munkres)
		link_list = input_ids.assign(linkage_cost = linkage_cost)
		winner_index = self.maximize(link_list)

		# Return boolean array
		winners = np.zeros(len(link_list), np.bool)
		winners[winner_index] = 1

		return winners, linkage_score

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
