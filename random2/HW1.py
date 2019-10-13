__author__ = 't.bagramyan'

def simulate( dt_start, dt_end, ls_symbols, alloc) :
 #skip 16 hours to close time
 dt_timeofday = dt.timedelta(hours=16)
 #timestamps for closeof every trading day
 ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)
 #read data
 c_dataobj = da.DataAccess('Yahoo')
 ls_keys = ['open' , 'high' , 'low' , 'close' , 'volume' , 'actual_close']
 ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
 d_data = dict(zip(ls_keys, ldf_data))
 #normalize prices
 normalized_price = d_data['close'].values/d_data['close'].values[0,:]
 #calculate portfolio value for every day
 portfolio = np.dot(normalized_price, alloc)
 #calculate cumulative return
 cum_ret = portfolio/portfolio[0]
 cumulative_return = cum_ret[-1]
 #calculate daily returns
 rets = portfolio.copy()
 tsu.returnize0(rets)
 #average daily return
 avg = np.average(rets)
 #volatility of daily returns
 volat = np.std(rets)
 #Sharpe ratio
 SR = np.sqrt(252)*avg/volat
 #plot result
 #plt.clf()
 #fig = plt.figure()
 #plt.plot(ldt_timestamps, rets)
 #plt.legend('ls_symbols')
 #plt.ylabel('Adjusted Close')
 #plt.xlabel('Date')
 #fig.autofmt_xdate(rotation=45)
 #plt.savefig('adjustedclose.pdf', format='pdf')
 
 return (SR, volat, avg, cumulative_return);

def Grid_optimizer( dt_start , dt_end , ls_symbols) :
 SR = 0	
 for h in range (0,11) :
	for k in range (0,11-h) :
		for l in range (0,11-h-k) :
			[SR_temp , volat_temp , avg_temp , cumulative_return_temp] = simulate (dt_start , dt_end , ls_symbols , [0.1*h , 0.1*k , 0.1*l , 1-0.1*(h+k+l)])
			if SR_temp>SR :
				SR = SR_temp
				volat = volat_temp
				avg  = avg_temp
				Opt_alloc =  [0.1*h , 0.1*k , 0.1*l , 1-0.1*(h+k+l)]
				cumulative_return = cumulative_return_temp	

 return (SR , volat , avg , Opt_alloc , cumulative_return)
 
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

[SR , volat , avg , Opt_alloc , cumulative_return] = Grid_optimizer(dt.datetime(2011,1,1) , dt.datetime(2011,12,31) , ['AAPL', 'GLD', 'GOOG', 'XOM'])

print ('Sharpe ratio' , SR)
print ('Volatility' , volat)
print ('Average return' , avg)
print ('Allocation' , Opt_alloc)
print ('Cumulative return' , cumulative_return)
