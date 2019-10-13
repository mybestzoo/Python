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

def datetrans( from_csv , to_csv ) :
# read csv
 orders = csv.reader(open(from_csv,'rU'),delimiter=',')
 #write the time-series to csv
 writer = csv.writer(open(to_csv,'wb'), delimiter=',')

 for row in orders:
	date = dt.datetime.strptime(row[0],"%m/%d/%Y")
	row_to_enter = [date.year , date.month , date.day , row[1] , row[2] , row[3]]
	writer.writerow(row_to_enter)	
 	 
# main

from_csv = str(sys.argv[1])
to_csv  = str(sys.argv[2])
 
datetrans( from_csv , to_csv )