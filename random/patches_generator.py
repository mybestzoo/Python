#import os 
import cv2
from numpy import*

# LOAD TRAINING AND TEST FRAMES
train_frame = cv2.imread('001.jpg')
test_frame = cv2.imread('010.jpg')

# set coordinates of the detected bounding box in both frames
train_r,h,train_c,w = 532,50,290,50
test_r,test_c, = 490,295

#--------------------------------------------------------------------------------------------------------------

# CREATE TRAINING SET from regions overlapping object and regions in background

N = 50 # number of samples
Xtrain = zeros((2*N,50,50,3))
Ytrain = zeros((2*N))
# first create positive samples iterating through pixels in roi
k = 0
for i in xrange(N/10):
	for j in xrange(N/10):
		Xtrain[k,:,:,:] = train_frame[train_c+(-1)**i*i:train_c+(-1)**i*i+w,train_r+(-1)**j*j:train_r+(-1)**j*j+h]
		Ytrain[k] = 1
		# show positive samples
		cv2.imshow("frame",train_frame[train_c+(-1)**i*i:train_c+(-1)**i*i+w,train_r+(-1)**j*j:train_r+(-1)**j*j+h])
		cv2.imwrite("data/train/object/car" + str(i) + str(j) + ".jpg", train_frame[train_c+(-1)**i*i:train_c+(-1)**i*i+w,train_r+(-1)**j*j:train_r+(-1)**j*j+h])
		key = cv2.waitKey(200) & 0xff
		k = k+1

# then create negative samples iterating through pixels in background
for i in xrange(N/10):
	for j in xrange(N/10):
		Xtrain[k,:,:,:] = train_frame[train_c+(-1)**i*(50+i):train_c+(-1)**i*(50+i)+w,train_r+(-1)**j*(50+j):train_r+(-1)**j*(50+j)+h]
		Ytrain[k] = 0
		# show negative samples
		cv2.imshow("frame",train_frame[train_c+(-1)**i*(50+i):train_c+(-1)**i*(50+i)+w,train_r+(-1)**j*(50+j):train_r+(-1)**j*(50+j)+h])
		cv2.imwrite("data/train/background/back" + str(i) + str(j) + ".jpg", train_frame[train_c+(-1)**i*(50+i):train_c+(-1)**i*(50+i)+w,train_r+(-1)**j*(50+j):train_r+(-1)**j*(50+j)+h])
		key = cv2.waitKey(200) & 0xff
		k = k+1
