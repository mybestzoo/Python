#! /usr/bin/env python
# -*- coding: utf-8 -*-

# import modules
import pandas as pd

def data_processing(df):
  # Load and preprocess data

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
 
# load excel table as dataframe
df = pd.read_excel(open('iiko.xlsx','rb'))

df = data_processing(df)

# group by date_time
#grouping = df.groupby(['Date_Time'])
#ts = grouping['Sum_after_discount'].sum()

ts = pd.Series(df['Sum_after_discount'].values, index = df['Date_Time'])

ts.to_csv('mini_store.csv')
