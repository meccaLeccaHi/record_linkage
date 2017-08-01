import pymysql
import pandas as pd

def test():
	print('Ni!')

def connect():
	''' Connect to mysql database '''
	global connection
	connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             db='current',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
	return connection

def exec_sql(sql):
    ''' Executes a sql command (string arg) and returns result as pandas dataframe '''
    return pd.read_sql(sql, connection)

def df_crossjoin(df1, df2, **kwargs):
    ''' Cross (or Cartesian)-joins two pandas dataframes '''
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    return_val = pd.merge(df1, df2, on='_tmpkey', **kwargs).drop('_tmpkey', axis=1)
    return_val.index = pd.MultiIndex.from_product((df1.index, df2.index))
    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)
    return return_val
