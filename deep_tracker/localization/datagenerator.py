import cv2
from numpy import*

def datagenerator(type,frame,N,(bb_r,h,bb_c,w)):
	# Create dataset from regions overlapping object and regions in background
	# type - 'train' for training set & 'test' for validation set
	# N - number of samples
	X = zeros((2*N,h,w,3))
	Y = zeros((2*N))
	# first create positive samples iterating through pixels in roi
	k = 0
	for i in xrange(N/10):
		for j in xrange(N/10):
			X[k,:,:,:] = frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h]
			Y[k] = 1
			# show positive samples
			cv2.imshow("frame",frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h])
			cv2.imwrite("data/"+type+"/object/car" + str(i) + str(j) + ".jpg", frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h])
			key = cv2.waitKey(200) & 0xff
			k = k+1
		
	X = zeros((2*N,50,50,3))
	Y = zeros((2*N))
	# then create negative samples iterating through pixels in background
	for i in xrange(N/10):
		for j in xrange(N/10):
			X[k,:,:,:] = frame[bb_c+(-1)**i*(50+i):bb_c+(-1)**i*(50+i)+w,bb_r+(-1)**j*(50+j):bb_r+(-1)**j*(50+j)+h]
			Y[k] = 0
			# show negative samples
			cv2.imshow("frame",frame[bb_c+(-1)**i*(50+i):bb_c+(-1)**i*(50+i)+w,bb_r+(-1)**j*(50+j):bb_r+(-1)**j*(50+j)+h])
			cv2.imwrite("data/"+type+"/background/back" + str(i) + str(j) + ".jpg", frame[bb_c+(-1)**i*(50+i):bb_c+(-1)**i*(50+i)+w,bb_r+(-1)**j*(50+j):bb_r+(-1)**j*(50+j)+h])
			key = cv2.waitKey(200) & 0xff
			k = k+1
	
# LOAD TRAINING AND TEST FRAMES
train_frame = cv2.imread('001.jpg')
test_frame = cv2.imread('010.jpg')

# set coordinates of the detected bounding box in both frames
train_r,h,train_c,w = 532,50,290,50
test_r,test_c, = 490,295

N = 100 # number of samples

# generate training set
datagenerator('train',train_frame,N,(532,50,290,50))
# generate validation set
datagenerator('validation',test_frame,N,(490,50,295,50))
