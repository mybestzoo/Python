Resources:
https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
http://michaelpaulschramm.com/simple-time-series-trend-analysis/

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import modules
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from __future__ import division

# Load and preprocess data

def data_processing(df):
	# delete Unnamed:6 column
	del df['Unnamed: 6']

	# rename columns
	df.columns = ['Date_Time', 'Check', 'Product_group', 'Product', 'Units', 'Sum_before_discount', 'Sum_after_discount', 'Net_cost', 'Markup']

	# delete rows containing всего
	df = df[~(df.Check.str.contains("всего") == False) ]

	# fill NaNs
	df = df.fillna(method='ffill')

	# convert Date_Time to datetime
	df['Date_Time'] = pd.to_datetime(df['Date_Time'])
	# add columns Date, Year, Months, Day, Hour
	df['Date'] = df['Date_Time'].dt.date
	df['Weekday'] = df['Date_Time'].dt.weekday
	df['Year'] = df['Date_Time'].dt.year
	df['Month'] = df['Date_Time'].dt.month
	df['Day'] = df['Date_Time'].dt.day
	df['Hour'] = df['Date_Time'].dt.hour

	# add columns Discount, Profit, Price
	df['Discount'] = df['Sum_before_discount'] - df['Sum_after_discount']
	df['Profit'] = df['Sum_after_discount']-df['Net_cost']
	df['Price'] = df['Sum_after_discount']/df['Units']

	# rearrange columns
	df = df [['Date_Time', 'Date', 'Weekday', 'Year', 'Month', 'Day', 'Hour', 'Check', 'Product_group', 'Product', 'Units', 'Sum_before_discount', 'Sum_after_discount', 'Net_cost', 'Price', 'Markup', 'Discount', 'Profit']]

	# Data Clearance
	df = df[(df['Units']>=0) & (df['Net_cost']>=0) & (df['Sum_before_discount']>=0) & (df['Sum_after_discount']>=0)]
	# drop March (not enough data)
	df = df[df['Month']>3]
	# drop August (not enough data)
	df = df[df['Month']<8]
	
	return df

	
	
# load excel table as dataframe and process it
df = pd.read_excel(open('iiko.xlsx','rb'))
df = data_processing(df)

# decompose data into current month and previous month
month_cur = df[df['Month'] == 7]
month_prev = df[df['Month'] == 6]

# calculate revenues
rev_cur = month_cur['Sum_after_discount'].sum()
rev_prev = month_prev['Sum_after_discount'].sum()

print 'Revenue in current  month:', rev_cur
print 'Revenue in previous  month:', rev_prev
print 'Revenue gain in this month compared to previous:', round((rev_cur-rev_prev)/rev_cur*100), '%'

# calculate revenues by product_groups
grouping_cur = month_cur.groupby(['Product_group'])
grouping_cur = grouping_cur['Sum_after_discount'].sum()
grouping_cur_pct = grouping_cur/rev_cur
grouping_prev = month_prev.groupby(['Product_group'])
grouping_prev = grouping_prev['Sum_after_discount'].sum()
grouping_prev_pct = grouping_prev/rev_prev

# ABC analysis
# sort product groups by percentage of revenues
ABC = grouping_prev_pct.sort_values(ascending=False)
# calculate cumulative sum
ABC = ABC.cumsum()
# decompose into groups: A 80% of revenue; B 80-95%; C 95%-100%.
A = ABC[ABC.values < 0.8].index
B = ABC[(ABC.values > 0.8) & (ABC.values < 0.95)].index
C = ABC[ABC.values > 0.95].index

print 'Group A 80 % revenue:' , A
print 'Group B 15 % revenue:' , B
print 'Group C 5 % revenue:' , C

# assign ABC labels to df
for group in A:
    df['ABC'].loc[df['Product_group'] == group] = 'A'
for group in B:
    df['ABC'].loc[df['Product_group'] == group] = 'B'
for group in C:
    df['ABC'].loc[df['Product_group'] == group] = 'C'
	
# again decompose data into current month and previous month
month_cur = df[df['Month'] == 7]
month_prev = df[df['Month'] == 6]

# calculate revenues by ABC groups
grouping_cur = month_cur.groupby(['ABC'])
grouping_cur = grouping_cur['Sum_after_discount'].sum()
grouping_cur_pct = grouping_cur/rev_cur
grouping_prev = month_prev.groupby(['ABC'])
grouping_prev = grouping_prev['Sum_after_discount'].sum()
grouping_prev_pct = grouping_prev/rev_prev

print 'Revenue gain in this month compared to previous by ABC groups:', (grouping_cur-grouping_prev)/grouping_cur*100, '%'
print 'Percentage of the total gain by Groups:', (grouping_cur-grouping_prev)/(rev_cur-rev_prev)*100, '%'
