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
import dlib
import os
import glob

cap = cv2.VideoCapture('video.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (960,720))

frameid = 0
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
			if int(fields[0]) == frameid and len(fields) == 1 and frameid%40 == 0:
				ListOfObjects = []
				print frameid
				j=i+1
				while len(lines[j])>4:
					l = lines[j].strip()
					f = l.split(" ")
					c,w,r,h = int(f[0]),int(f[2])-int(f[0]),int(f[1]),int(f[3])-int(f[1])
					cv2.rectangle(frame, (c, r), (c+w, r+h), (0,255,0), 3)
					tracker = dlib.correlation_tracker()
					tracker.start_track(frame, dlib.rectangle(c, r, c+w, r+h))
					ListOfObjects.append(tracker)
					j += 1
		
		for tracker in ListOfObjects:
			tracker.update(frame)
			# Draw it on image
			x = tracker.get_position()
			cv2.rectangle(frame, (int(x.left()),int(x.top())), (int(x.right()),int(x.bottom())), (0,0,255),2)	
		
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
