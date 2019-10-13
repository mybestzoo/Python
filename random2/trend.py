#! /usr/bin/env python
# -*- coding: utf-8 -*-

#Resources:
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#http://michaelpaulschramm.com/simple-time-series-trend-analysis/
#http://pandas.pydata.org/pandas-docs/version/0.9.0/computation.html

"""
resample how:
B       business day frequency
C       custom business day frequency (experimental)
D       calendar day frequency
W       weekly frequency
M       month end frequency
BM      business month end frequency
CBM     custom business month end frequency
MS      month start frequency
BMS     business month start frequency
CBMS    custom business month start frequency
Q       quarter end frequency
BQ      business quarter endfrequency
QS      quarter start frequency
BQS     business quarter start frequency
A       year end frequency
BA      business year end frequency
AS      year start frequency
BAS     business year start frequency
BH      business hour frequency
H       hourly frequency
T       minutely frequency
S       secondly frequency
L       milliseonds
U       microseconds
N       nanoseconds
"""

# import modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def trend_pattern(data_path):
	# load time series
	ts = pd.Series.from_csv(data_path)

	# resample to weeks
	ts_W = ts.resample('W').sum()
	#print ts_W.pct_change()

	# resample to months
	ts_M = ts.resample('M').sum()
	#print ts_M.pct_change()

	# resample to quater
	ts_Q = ts.resample('Q').sum()
	#print ts_Q.pct_change()

	# resample to annual
	ts_A = ts.resample('A').sum()
	#print ts_A.pct_change()

	"""


	#fix nans
	ts[np.isnan(ts)] = 0

	#Determing rolling statistics
	rolmean = ts.rolling(window=2).mean()

	#Plot rolling statistics:
	orig = plt.plot(ts, color='blue',label='Original')
	mean = plt.plot(rolmean, color='red', label='Rolling Mean')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show()#(block=False)

	# Test the trend
	trend = rolmean.values
	trend = trend[np.isfinite(rolmean.values)]
	test_trend,h,p,z = mk_test(trend,alpha=0.05)
	print test_trend, h 

	#Decompose season
	from statsmodels.tsa.seasonal import seasonal_decompose
	decomposition = seasonal_decompose(ts.values, freq = 4, model = 'additive')
	#Plot seasonal pattern
	pattern = plt.bar(range(5,10), decomposition.seasonal[5:10]-np.min(decomposition.seasonal[5:10]))
	plt.show()

	#Calculate returns
	daily_return = ts.pct_change(1) # 1 for ONE DAY lookback
	monthly_return = ts.pct_change(4) # 21 for ONE MONTH lookback
	annual_return = ts.pct_change(8) # 252 for ONE YEAR lookback

	annual_return.plot()
	plt.show()

	"""

trend_pattern('mini_store.csv')