import operator
import csv
import pandas as pd
import numpy as np
import math
import copy
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

"""
Accepts a list of symbols along with start and end date
Returns the csv file with orders:
When an event occurs, buy 100 shares of the equity on that day.
Sell automatically 5 trading days later.
"""
def bollinger(price,period):
    SMA = pd.rolling_mean(price,period,min_periods=period)
    STD = pd.rolling_std(price,period,min_periods=period)
    # bollinger bands
    upper = SMA+STD
    lower = SMA-STD
    # bollinger indicator
    bollinger_ind = (price-SMA)/STD
    return bollinger_ind

def find_events(ls_symbols, d_data):
	''' Finding the event dataframe '''
	df_close = d_data['close']
	ts_market = df_close['SPY']

	print "Finding Events"

	# Creating an empty dataframe
	df_events = copy.deepcopy(df_close)
	df_events = df_events * np.NAN

	# Time stamps for the event range
	ldt_timestamps = df_close.index

	#calculate rolling statistics
	indicator = bollinger(df_close,20)
	indicator_market = bollinger(ts_market,20)
	
	# create csv file for trade orders
	writer = csv.writer(open('TradeOrders.csv','wb'), delimiter=',')	
	for s_sym in ls_symbols:
		for i in range(1, len(ldt_timestamps)):
			# Event is found if the symbol indicator hits the -2.0 value while the
            # market indicator is higher then 1.0
			if indicator[s_sym].ix[ldt_timestamps[i]] < -2.0 and indicator[s_sym].ix[ldt_timestamps[i-1]] >= -2.0 and indicator_market[ldt_timestamps[i]] >= 1.4:
				buy_row = [ldt_timestamps[i].year, ldt_timestamps[i].month , ldt_timestamps[i].day  , s_sym, 'Buy', 100]
				sell_row = [ldt_timestamps[min(i+5,len(ldt_timestamps)-1)].year , ldt_timestamps[min(i+5,len(ldt_timestamps)-1)].month , ldt_timestamps[min(i+5,len(ldt_timestamps)-1)].day  , s_sym, 'Sell', 100]
				writer.writerow(buy_row)
				writer.writerow(sell_row)
			
	return df_events


if __name__ == '__main__':
    dt_start = dt.datetime(2008, 1, 1)
    dt_end = dt.datetime(2009, 12, 31)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))

    dataobj = da.DataAccess('Yahoo')
    ls_symbols = dataobj.get_symbols_from_list('sp5002012')
    ls_symbols.append('SPY')

    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)

    df_events = find_events(ls_symbols, d_data)
    print "Creating trade orders"
