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
		
	def prec_recall(self, linkage_score, theta_range, true_matches):
		''' 
		Get precision and recall from linkage data on each iteration
		- Precision: the percent of pairs with a score above theta that are real matches
		- Recall: the percent of known matched pairs that get a score above theta
		'''
		precision = []
		recall = []
		
		theta = np.percentile(linkage_score.loc[true_matches],theta_range)

		for cutoff in theta:
			
			winners = (linkage_score>=cutoff)
			#winners = (big_bool['linkage_score']>cutoff)&big_bool['pair_match']

			# - numerator: number of NOMINATED pairs with a score above theta that are real matches
			real_picks = sum(winners&true_matches)
			# - denominater: number of NOMINATED pairs with a score above theta
			picks = sum(winners)
			precision.append(real_picks/float(picks)*100)

			# - numerator: number of NOMINATED pairs with a score above theta that are real matches
			# - denominater: number of pairs that are real matches
			real_matches = true_matches.sum()
			recall.append(real_picks/float(real_matches)*100)
					
		self.precision_list.append(precision)
		self.recall_list.append(recall)	
		self.iter_qual_list.append(np.nanmean(precision) + np.nanmean(recall))	



