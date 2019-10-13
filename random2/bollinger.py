import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.DataAccess as da
import numpy as np

"""
Calculates the bollinger bands

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

# main

# set start date and end date + lookup period for rolling mean
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2010, 12, 31)
period = 20
timestamps = du.getNYSEdays(start, end, dt.timedelta(hours=16))

# get prices
data = da.DataAccess('Yahoo')
symbol = 'MSFT'
price = data.get_data(timestamps,[symbol],'close')

# calculate rolling statistics
SMA = pd.rolling_mean(price,period,min_periods=period)
STD = pd.rolling_std(price,period,min_periods=period)

# bollinger bands
upper = SMA+STD
lower = SMA-STD

# bollinger indicator
bollinger_ind = bollinger(price,20)

print(bollinger_ind[str(dt.date(2010, 6, 23))])

#plot 
#plt.clf()
fig = plt.figure(1)
plt.subplot(211)
plt.plot(timestamps, price,
		 timestamps, SMA,
		 timestamps, upper,
		 timestamps, lower)
#plt.fill_between(timestamps,lower,upper,where=np.isfinite(lower))
plt.legend([symbol,'SMA'])
plt.ylabel('Adjusted Close')
plt.xlabel('Date')

plt.subplot(212)
plt.plot(timestamps, bollinger_ind)
plt.fill_between(timestamps,-np.ones(len(timestamps)),np.ones(len(timestamps)),facecolor='blue', alpha=0.5)
fig.autofmt_xdate(rotation=45)
plt.savefig('bollinger.pdf', format='pdf')