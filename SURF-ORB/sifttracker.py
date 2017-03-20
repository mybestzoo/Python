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
import time

class Object:
	
	def __init__(self, track_window = 0, image = 0):
		self.track_window = track_window
		self.image = image
		
	def setup(self,r,h,c,w):
		self.track_window = (c,r,w,h)
		self.image =  cv2.cvtColor(frame[r:r+h, c:c+w],cv2.COLOR_BGR2GRAY)
			
	def detect(self,frame,out_frame):
		(c,r,w,h) = self.track_window
		img1 = self.image 											# queryImage
		roi = 0 # region of interest size
		img2 = cv2.cvtColor(frame[r-roi:r+h+roi, c-roi:c+w+roi],cv2.COLOR_BGR2GRAY) 				# trainImage

		# Initiate SIFT detector
		sift = cv2.SIFT()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)
		
		# create BFMatcher object
		bf = cv2.BFMatcher()
		
		# Match descriptors
		matches = bf.knnMatch(des1,des2, k=2)
		
		# Apply ratio test
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append(m)

		# For each pair of points we have between both images draw circles
		# and calculate the bounding box
		rows2 = out_frame.shape[0]
		cols2 = out_frame.shape[1]
		a = cols2
		b = rows2
		e = 0
		f = 0
		for mat in good:

			# Get the matching keypoints
			img2_idx = mat.trainIdx

			# x - columns
			# y - rows
			(x2,y2) = kp2[img2_idx].pt
			
			# Draw a small circle at coordinates
			# radius 4
			# colour blue
			# thickness = 1
			cv2.circle(out_frame, (c-roi+int(x2),r-roi+int(y2)), 4, (255, 0, 0), 1)

			# Find coordinates of the bounding box
			if  x2 < a :
				a = int(x2)
			if	x2 > e :
				e = int(x2)
			if y2 < b :
				b = int(y2)
			if y2 > f :
				f = int(y2)
				
		# Draw a bounding box
		cv2.rectangle(out_frame, (c-roi+a,r-roi+b), (c-roi+e,r-roi+f), 255,2)
		
		# Reassign the object position
		#self.track_window = (c,r,w,h)
		
		# Show the image
		cv2.imshow('Matched Features', out_frame)
			
cap = cv2.VideoCapture('film1.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1288,728))

frameid = 1
ListOfObjects = []
while(1):
	ret ,frame = cap.read()
	
	if ret == True:
		
		frameid += 1
	
		# parse file and extract measurements
		infile = open("boxes1.txt", "r")
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
		
		out_frame = frame
		tic = time.time()
		for ob in ListOfObjects:
			ob.detect(frame,out_frame)
		tac = time.time()
		print(tac-tic)
		
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
