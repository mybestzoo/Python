from PIL import Image, ImageDraw
from numpy import *
import matplotlib.pyplot as plt

def mono(rgb):
	r, g, b = rgb[:,:,0]/255, rgb[:,:,1]/255, rgb[:,:,2]/255
	mono = (r + g + b) / 3
	return mono

def showIm(I):
	img = Image.new('1',(shape(I)[0],shape(I)[1]))
	pixels = img.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			pixels[i, j] = I[j,i]
	img.show()	

def index(I,i,j):
	v = zeros(8)
	if I[i,j+1] == 0:
		v[0] = 1
		v[1] = 1
		v[4] = 1
	if I[i+1,j+1] == 0:
		v[1] = 1
	if I[i+1,j] == 0:
		v[1] = 1
		v[2] = 1
		v[5] = 1
	if I[i+1,j-1] == 0:
		v[2] = 1
	if I[i,j-1] == 0:
		v[2] = 1
		v[3] = 1
		v[6] = 1
	if I[i-1,j-1] == 0:
		v[3] = 1
	if I[i-1,j] == 0:
		v[3] = 1
		v[0] = 1
		v[7] = 1
	if I[i-1,j+1] == 0:
		v[0] = 1
	ind = v[0]+v[1]+v[2]+v[3]-v[4]-v[5]-v[6]-v[7]
	return ind
	
#read image
img = Image.open('binary.png')
rgb = array(img)

#convert to mono
I = mono(rgb)

l = 0
n = 0

while l< 1:
	for i in arange(1,img.size[1]-2):
		for j in arange(1,img.size[0]-2):
			if I[i,j] == 0 and index(I,i,j) == 1:
				I[i,j] = 1
				n = n+1
	print(n)
	if n == 0:
		l = 10
	else: 
		n=0	
	

showIm(I)