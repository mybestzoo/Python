import cv2
from numpy import*

def datagenerator(type,object_num,frame,N,(bb_r,h,bb_c,w)):
	# Create dataset from regions overlapping object and regions in background
	# object_num - number of object
	# type - 'test' or 'validation'
	# N - number of samples
	X = zeros((2*N,h,w,3))
	# first create positive samples iterating through pixels in roi
	k = 0
	for i in xrange(N/10):
		for j in xrange(N/10):
			X[k,:,:,:] = frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h]
			# show positive samples
			cv2.imshow("frame",frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h])
			cv2.imwrite("data/" + type + "/" + object_num + "/" + object_num + str(i) + str(j) + ".jpg", frame[bb_c+(-1)**i*i:bb_c+(-1)**i*i+w,bb_r+(-1)**j*j:bb_r+(-1)**j*j+h])
			key = cv2.waitKey(200) & 0xff
			k = k+1
	
# LOAD TRAINING AND TEST FRAMES
train_frame = cv2.imread('001.jpg')
test_frame = cv2.imread('010.jpg')

# set coordinates of the detected bounding box in both frames
#train1_r,h1,train1_c,w1 = 532,50,290,50
#train2_r,h2,train2_c,w2 = 620,260,290,260
#train3_r,h3,train3_c,w3 = 880,80,300,80
#train4_r,h4,train4_c,w4 = 980,190,260,190

#test1_r,test1_c, = 490,295
#test2_r,test2_c, = 620,290
#test1_r,test1_c, = 880,300
#test1_r,test1_c, = 980,260

N = 100 # number of samples

# generate training set
datagenerator('train','object1',train_frame,N,(532,50,290,50))
datagenerator('train','object2',train_frame,N,(620,260,290,260))
datagenerator('train','object3',train_frame,N,(880,80,300,80))
datagenerator('train','object4',train_frame,N,(980,190,260,190))

# generate validation set
datagenerator('validation','object1',test_frame,N,(490,50,295,50))
datagenerator('validation','object2',test_frame,N,(620,260,290,260))
datagenerator('validation','object3',test_frame,N,(880,80,300,80))
datagenerator('validation','object4',test_frame,N,(980,190,260,190))
