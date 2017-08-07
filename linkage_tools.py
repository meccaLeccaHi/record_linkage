import pymysql
import pandas as pd
import numpy as np
from munkres import Munkres

class Linker(Munkres):
	def __init__(self):
		self.connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             db='current',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
		#self.munkres = Munkres()

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

	def maximize(self, link_list):
		''' Maximize linkage pairing with Munkres algo '''
		link_cost_mat = link_list.pivot(index='newb_id',columns='bc_id').fillna(0)

		# Maximize pairing scores (minimize cost with Munkres)
		indexes = self.compute(list(link_cost_mat.as_matrix()))

		nom_newb_id = list(link_cost_mat['linkage_cost'].index) # Get row names
		nom_bc_id = link_cost_mat['linkage_cost'].columns.tolist() # Get column names

		return np.concatenate([np.where((link_list['newb_id']==nom_newb_id[row])&
                                                 (link_list['bc_id']==nom_bc_id[col])) 
                                        for row,col in indexes]).ravel()
		

