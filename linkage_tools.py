import pymysql
import pandas as pd
import numpy as np
from munkres import Munkres
import socket

class Linker(Munkres):
	def __init__(self):

		# Initialize connection to sql server
		if socket.gethostname()=='adam-server':
			host = 'localhost'
		else:
			host = '127.0.0.1'
		
		self.connection = pymysql.connect(host=host,
							user='root',
							db='mdc_2017_09_07', 
							charset='utf8mb4',
							cursorclass=pymysql.cursors.DictCursor) 

#db= 'current', db='mdc_2017_08_11', db='mdc_2017_09_05', 

	def exec_sql(self, sql):
		''' Executes a sql command (string arg) and returns result as pandas dataframe '''
		return pd.read_sql(sql, self.connection)

	# @ staticmethod
	def df_crossjoin(self, df1, df2, **kwargs):
		''' Cross (or Cartesian)-joins two pandas dataframes '''
		df1['_tmpkey'] = 1
		df2['_tmpkey'] = 1
		return_val = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
		return_val.index = pd.MultiIndex.from_product((df1.index, df2.index))
		df1.drop('_tmpkey', axis=1, inplace=True)
		df2.drop('_tmpkey', axis=1, inplace=True)
		return return_val

	def digitize_series(self, vals, bins):
		''' Return numpy.digitize output as pandas Series
			Example usage: digitize_series(cross_table['birth_weight'],bw_bins) '''
		return(pd.Series(np.digitize(list(vals),bins)))

	def maximize(self, link_list):
		''' Maximize linkage pairing with Munkres algo '''
		link_cost_mat = link_list.pivot(index='newb_id',columns='bc_id').fillna(0)

		# Maximize pairing scores (minimize cost with Munkres)
		indexes = self.compute(list(link_cost_mat.as_matrix()))

		nom_newb_id = list(link_cost_mat['linkage_cost'].index) # Get row names
		nom_bc_id = link_cost_mat['linkage_cost'].columns.tolist() # Get column names

		max_ids = [np.where((link_list['newb_id']==nom_newb_id[row])&
                                                 (link_list['bc_id']==nom_bc_id[col])) 
                                        for row,col in indexes]

		return np.concatenate(max_ids).ravel()
		
	def test_classifier(self, classifier_matches, true_matches):

		match_guess = np.where(classifier_matches)[0]
		match_answer = np.where(true_matches)[0]
		no_match_guess = np.where(np.logical_not(classifier_matches))[0]
		no_match_answer = np.where(np.logical_not(true_matches))[0]

		'''
		True positives. These are the record pairs that have been classified as matches and
		that are true matches. These are the pairs where both records refer to the same
		entity.
		'''
		TP = sum(match_guess==match_answer)

		'''
		False positives. These are the record pairs that have been classified as matches,
		but they are not true matches. The two records in these pairs refer to two different
		entities. The classifier has made a wrong decision with these record pairs. These
		pairs are also known as false matches.
		'''
		FP = sum(match_guess!=match_answer)

		'''
		True negatives. These are the record pairs that have been classified as non-matches,
		and they are true non-matches. The two records in pairs in this category do refer
		to two different real-world entities.
		'''
		TN = sum(no_match_guess==no_match_answer)

		'''
		False negatives. These are the record pairs that have been classified as non-matches,
		but they are actually true matches. The two records in these pairs refer to the same
		entity. The classifier has made a wrong decision with these record pairs. These
		pairs are also known as false non-matches.
		'''
		FN = sum(no_match_guess!=no_match_answer)
		
		'''
		Precision: proportion of how many of the classified matches (TP + FP) have been correctly
		classified as true matches (TP) a.k.a. positive-predictive value
		'''
		precision = TP/float(TP + FP)

		'''
		Recall: measures the proportion of true matches (TP + FN) that have been classified correctly 			(TP). It thus measures how many of the actual true matching record pairs have been correctly
		classified as matches. a.k.a. true-positive rate or sensitivity
		'''
		recall = TP/float(TP + FN)

		'''
		F-measure: calculates the harmonic mean between precision and recall. The f-measure combines
		precision and recall and only has a high value if both precision and recall are high. Aiming
		to achieve a high f-measure requires to find the best compromise between precision and recall.
		'''
		fscore = 2*((precision*recall)/(precision+recall))

		'''
		# - numerator: number of NOMINATED pairs with a score above theta that are real matches
		real_picks = sum(winners&true_matches)
		# - denominater: number of NOMINATED pairs with a score above theta
		picks = sum(winners)
		precision.append(real_picks/float(picks)*100)

		# - numerator: number of NOMINATED pairs with a score above theta that are real matches
		# - denominater: number of pairs that are real matches
		real_matches = true_matches.sum()
		recall.append(real_picks/float(real_matches)*100)
		'''
			
		self.precision_list.append(precision)
		self.recall_list.append(recall)	
		self.fscore_list.append(fscore)



