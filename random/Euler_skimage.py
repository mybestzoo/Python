from PIL import Image, ImageDraw
from numpy import *
from skimage.measure import regionprops

def mono(rgb):
	r, g, b = rgb[:,:,0]/255, rgb[:,:,1]/255, rgb[:,:,2]/255
	mono = (r + g + b) / 3
	return mono	

#read image
img0 = Image.open('binary.png')
rgb = array(img0)

#convert to mono
I = mono(rgb)

for region in regionprops(I):
	print "Euler number: "
	print region.euler_number
	print ""
