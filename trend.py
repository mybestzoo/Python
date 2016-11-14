#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Resources:
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#http://michaelpaulschramm.com/simple-time-series-trend-analysis/
#http://pandas.pydata.org/pandas-docs/version/0.9.0/computation.html

# import modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def trend(data_path):
	# load time series
	ts = pd.Series.from_csv(data_path)

	# resample to weeks
	ts_W = ts.resample('W').sum()
	# calculate percentage change
	ts_W_pct = ts_W.pct_change()

	# resample to months
	ts_M = ts.resample('M').sum()
	# calculate percentage change
	ts_M_pct = ts_M.pct_change()
    	
	# resample to quarters
	ts_Q = ts.resample('Q').sum()
	# calculate percentage change
	ts_Q_pct = ts_Q.pct_change()    
    	
	# resample to annual
	ts_A = ts.resample('A').sum()
	# calculate percentage change
	ts_A_pct = ts_A.pct_change()
   	
	ts_Q.plot()
	plt.show()
	
	ts_M.plot()
	plt.show()
	
	ts_W.plot()
	plt.show()
	
	print 'Results for this week compared to the previous:', round(ts_W_pct.iloc[-1]*100), '%'
	print 'Results for this month compared to the previous:', round(ts_M_pct.iloc[-1]*100), '%'
	print 'Results for this quarter compared to the previous:', round(ts_Q_pct.iloc[-1]*100), '%'
	print 'Results for this year compared to the previous:', round(ts_A_pct.iloc[-1]*100), '%'
	
trend_pattern('mini_store.csv')
