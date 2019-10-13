import sys
import csv
import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import time
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

"""
Accepts an amount of start cash along iwth csv file with orders
Returns the csv file with cumulative value of portfolio

"""

def marketsim( start_cash, orders_file ) :
#read csv
 orders = csv.reader(open(orders_file,'rU'),delimiter=',')
 #create two lists for all dates and symbols
 symbols = []
 dates = []
 for row in orders:
	dates.append(dt.datetime(int(row[0]),int(row[1]),int(row[2])))
	symbols.append(row[3])
 #remove duplicates
 symbols = list(set(symbols))
 dates = list(set(dates))
 #set the data provider to Yahoo
 database = da.DataAccess('Yahoo')
 #timestamps for close of every trading day ?: add +dt.timedelta(days=1)
 timestamps = du.getNYSEdays( min(dates) , max(dates)+dt.timedelta(days=1) , dt.timedelta(hours=16) )
 #read actual close price for symbols on dates
 price = database.get_data(timestamps,symbols,'close')
 for s_key in symbols:
        price[s_key] = price[s_key].fillna(method='ffill')
        price[s_key] = price[s_key].fillna(method='bfill')
        price[s_key] = price[s_key].fillna(1.0)
 #create trade matrix
 trade_matrix = pd.DataFrame( 0 , index = timestamps , columns = symbols)
 #create cash timeseries with start_cash initial
 cash = pd.Series( 0 , index = timestamps , name = 'Cash')
 cash[0] = float(start_cash)
 #fill trade matrix with number of traded shares and cash with cash used in trades
 orders = csv.reader(open(orders_file,'rU'),delimiter=',')
 for row in orders:
	date = str(dt.date(int(row[0]),int(row[1]),int(row[2])))
	if row[4] == 'Buy':
		i = 1.0
	else:
		i = -1.0	
	trade_matrix.loc[date,row[3]] = trade_matrix.loc[date,row[3]] + i*float(row[5])
	cash[date] = cash[date] - i*float(row[5])*float(price.loc[date,row[3]])
 #append cash to price
 price['Cash'] = 1.0
 trade_matrix['Cash'] = cash
 #use cummulative sum to convert the trade matrix into holding matrix
 trade_matrix = np.cumsum(trade_matrix)
 #multiply price to trade_matrix to get holdings
 fund = pd.Series(np.diag(np.dot(trade_matrix,price.transpose())), index = timestamps , name = 'Fund')
 #with pd.option_context('display.max_rows', 999, 'display.max_columns', 999):
 #   print(price['2008-02-25'])
 #write the time-series to csv
 writer = csv.writer(open('portfolio.csv','wb'), delimiter=',')
 for row_index in fund.index:
	 row_to_enter = [row_index.year,row_index.month,row_index.day,fund[row_index]]
	 writer.writerow(row_to_enter)
 return ;
 	 

start_cash = sys.argv[1]
orders_file  = sys.argv[2]

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)
 
marketsim( start_cash, orders_file )
