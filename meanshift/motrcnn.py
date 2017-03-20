# http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html#gsc.tab=0

# Video 2039
#r1,h1,c1,w1 = 290,50,400,120
#r2,h2,c2,w2 = 300,50,20,100

# Video 2653
#r1,h1,c1,w1 = 300,70,400,120
#r2,h2,c2,w2 = 290,70,600,120

# Video 1721
#r1,h1,c1,w1 = 300,30,630,60
#r2,h2,c2,w2 = 300,30,700,60

import numpy as np
import cv2

class Object:
	
	def __init__(self, track_window = 0, roi_hist = 0, u = np.array([0.1, 0.1]), P = np.diag((0.01, 0.01, 0.01, 0.01))):
		self.track_window = track_window
		self.roi_hist = roi_hist
		self.u = u
		self.P = P
	
	def setup(self,r,h,c,w):
		self.track_window = (c,r,w,h)
		# set up the ROI for tracking
		roi = frame[r:r+h, c:c+w]
		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
		self.roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	
	def Kalman(self,Y=0):
		# Parameters of the Kalman filter
		dt = 0.1
		A = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])
		B = np.eye(4)
		H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
		Q = np.array([[pow(dt,4)/float(4), 0, pow(dt,3)/float(2) , 0], [0, pow(dt,4)/float(4), 0, pow(dt,3)/float(2)], [pow(dt,3)/float(2), 0, pow(dt,2), 0], [0, pow(dt,3)/float(2), 0,pow(dt,2)]])
		R = np.array([[0.1, 0], [0, 0.1]])
		U = np.zeros((4,1))
		
		# predict
		X = np.array([[self.track_window[0]], [self.track_window[1]], [self.u[0]], [self.u[1]]])
		X = np.dot(A, X) + np.dot(B, U)
		self.P = np.dot(A, np.dot(self.P, A.T)) + Q
		
		# update
		IM = np.dot(H, X)
		IS = R + np.dot(H, np.dot(self.P, H.T))
		K = np.dot(self.P, np.dot(H.T, np.linalg.inv(IS)))
		X = X + np.dot(K, (Y-IM))
		self.P = self.P - np.dot(K, np.dot(IS, K.T))
		self.track_window = (int(X[0]), int(X[1]),self.track_window[2],self.track_window[3])
		self.u = (X[2],X[3])
		
		# Draw it on image meanshift
		x,y,w,h = self.track_window
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)	
				
	def detect(self,hsv):
	
		# Setup the termination criteria, either 10 iteration or move by at least 1 pt
		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

		dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)

		# apply meanshift to get the new location
		ret, Y = cv2.meanShift(dst, self.track_window, term_crit)
		Y = [[Y[0]],[Y[1]]]
		
		# Draw it on image
		#x,y,w,h = Y
		#cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
		return Y
		
	def track(self,Y=0):
		if self.track_window[0] >= 0 and self.track_window[1] >= 0 and self.track_window[0]+self.track_window[2] <= 1288 and self.track_window[1]+self.track_window[3] <= 728:
			self.Kalman(self.detect(Y))
			#self.detect(Y)
			
cap = cv2.VideoCapture('video.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (960,720))

frameid = 1
ListOfObjects = []
while(1):
	ret ,frame = cap.read()
	
	if ret == True:
		
		frameid += 1
	
		# parse file and extract measurements
		infile = open("boxess.txt", "r")
		lines = infile.readlines()
		for i in xrange(len(lines)):
			line = lines[i].strip()
			fields = line.split(" ")
			if int(fields[0]) == frameid and len(fields) == 1 and frameid%10 == 0:
				ListOfObjects = []
				print frameid
				j=i+1
				while len(lines[j])>4:
					l = lines[j].strip()
					f = l.split(" ")
					c,w,r,h = int(f[0]),int(f[2])-int(f[0]),int(f[1]),int(f[3])-int(f[1])
					cv2.rectangle(frame, (c, r), (c+w, r+h), (0,255,0), 3)
					car = Object()
					car.setup(r,h,c,w)
					ListOfObjects.append(car)
					j += 1
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		
		for ob in ListOfObjects:
			ob.track(hsv)
			
		cv2.imshow("frame",frame)
		
		# write the frame
		out.write(frame)
		
		key = cv2.waitKey(50) & 0xff
	
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break
	else:
		break
		
cv2.destroyAllWindows()
cap.release()
