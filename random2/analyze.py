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
import matplotlib.pyplot as plt

"""
Accepts a csv file with portfolio values and a benchmark symbol
Returns the statistics of the portfolio and plots a graph
python analyze.py values.csv \$SPX

"""

def analyze( values_file , bench_symbol ) :
# read csv
 orders = csv.reader(open(values_file,'rU'),delimiter=',')
 # create two lists for all values and dates
 values = []
 dates = []
 for row in orders:
	dates.append(dt.datetime(int(row[0]),int(row[1]),int(row[2]),16,0,0))
	values.append(float(row[3]))	
 # download prices for benchmark	
 # set the data provider to Yahoo
 database = da.DataAccess('Yahoo')
 # read adjucted close price for bench_symbol on dates
 bench_price = database.get_data(dates,[bench_symbol],'close')
 # normalize prices
 bench_price[bench_symbol] = bench_price.values/bench_price.values[0]
 values[:] = [x / values[0] for x in values]
 # append bench price and portfolio values
 bench_price['Portfolio'] = values
 #plot normalized prices
 plt.clf()
 fig = plt.figure()
 plt.plot(dates, bench_price)
 plt.legend([bench_symbol,'Portfolio'])
 plt.ylabel('Normalized Adjusted Close')
 plt.xlabel('Date')
 fig.autofmt_xdate(rotation=45)
 plt.savefig('adjustedclose.pdf', format='pdf')
 # calculate daily returns
 rets_portfolio = bench_price['Portfolio'].copy()
 tsu.returnize0(rets_portfolio)
 rets_bench = bench_price[bench_symbol].copy()
 tsu.returnize0(rets_bench)
 #average daily return
 avg_portfolio = np.average(rets_portfolio)
 avg_bench = np.average(rets_bench)
 #volatility of daily returns
 volat_portfolio = np.std(rets_portfolio)
 volat_bench = np.std(rets_bench)
 #Sharpe ratio
 SR_portfolio = np.sqrt(252)*avg_portfolio/volat_portfolio
 SR_bench = np.sqrt(252)*avg_bench/volat_bench

 print( 'Data range: %s to % s' %(dt.datetime.strftime(dates[0],"%Y-%m-%d") , dt.datetime.strftime(dates[-1],"%Y-%m-%d") ))
 print('')
 print( 'Sharpe ratio of portfolio: %s' ) %SR_portfolio
 print( 'Sharpe ratio of %s: %s' ) %(bench_symbol,SR_bench)
 print('')
 print( 'Total return of portfolio: %s ' ) %values[-1]
 print( 'Total return of %s: %s' ) %(bench_symbol,bench_price[bench_symbol].ix[-1])
 print('')
 print( 'Standard deviation of portfolio: %s' ) %volat_portfolio
 print( 'Standard deviation of %s: %s' ) %(bench_symbol,volat_bench)
 print('')
 print( 'Average daily return of portfolio: %s' ) %avg_portfolio
 print( 'Average daily return of %s: %s' ) %(bench_symbol,avg_bench)
 return ;
 	 
# main

values_file = sys.argv[1]
bench_symbol  = sys.argv[2]
 
analyze( values_file , bench_symbol )
