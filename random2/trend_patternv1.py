#! /usr/bin/env python
# -*- coding: utf-8 -*-

#Resources:
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#http://michaelpaulschramm.com/simple-time-series-trend-analysis/
#http://pandas.pydata.org/pandas-docs/version/0.9.0/computation.html

# import modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def mk_test(x, alpha = 0.05):  
    # Analyze trend by Mann-Kendall test
	from scipy.stats import norm, mstats
	"""
	Input:
		x:   a vector of data
		alpha: significance level (0.05 default)

	Output:
		trend: tells the trend (increasing, decreasing or no trend)
		h: True (if trend is present) or False (if trend is absence)
		p: p value of the significance test
		z: normalized test statistics 
	"""
	n = len(x)

	# calculate S 
	s = 0
	for k in range(n-1):
		for j in range(k+1,n):
			s += np.sign(x[j] - x[k])

	# calculate the unique data
	unique_x = np.unique(x)
	g = len(unique_x)

	# calculate the var(s)
	if n == g: # there is no tie
		var_s = (n*(n-1)*(2*n+5))/18
	else: # there are some ties in data
		tp = np.zeros(unique_x.shape)
		for i in range(len(unique_x)):
			tp[i] = sum(unique_x[i] == x)
		var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18

	if s>0:
		z = (s - 1)/np.sqrt(var_s)
	elif s == 0:
			z = 0
	elif s<0:
		z = (s + 1)/np.sqrt(var_s)

	# calculate the p_value
	p = 2*(1-norm.cdf(abs(z))) # two tail test
	h = abs(z) > norm.ppf(1-alpha/2) 

	if (z<0) and h:
		trend = 'decreasing trend'
	elif (z>0) and h:
		trend = 'increasing trend'
	else:
		trend = 'no trend'

	return trend, h, p, z

def trend_pattern(data_path):
	# load time series
	ts = pd.Series.from_csv(data_path)

	# resample to weeks
	ts = ts.resample('w').sum()
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

trend_pattern('mini_store.csv')