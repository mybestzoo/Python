#dataset texures http://www.cb.uu.se/~gustaf/texture/

#from __future__ import division
import scipy.stats as stats
from PIL import Image, ImageDraw
from numpy import *
import matplotlib.pyplot as plt
"""
def showIm(I):
	img = Image.new('1',(shape(I)[0],shape(I)[1]))
	pixels = img.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			pixels[i, j] = I[j,i]
	img.show()	

def index(I,i,j,c):
# calulate the index of pixel. Parameter c stands for index corresponding to image (c = 1) or background (c = 0)
	v = zeros(8)
	k = 0
	if I[i,j+1] == c:
		v[0] = 1
		v[1] = 1
		v[4] = 1
		k += 1
	if I[i+1,j+1] == c:
		v[1] = 1
		k += 1
	if I[i+1,j] == c:
		v[1] = 1
		v[2] = 1
		v[5] = 1
		k += 1
	if I[i+1,j-1] == c:
		v[2] = 1
		k += 1
	if I[i,j-1] == c:
		v[2] = 1
		v[3] = 1
		v[6] = 1
		k += 1
	if I[i-1,j-1] == c:
		v[3] = 1
		k += 1
	if I[i-1,j] == c:
		v[3] = 1
		v[0] = 1
		v[7] = 1
		k += 1
	if I[i-1,j+1] == c:
		v[0] = 1
		k += 1
	if k == 0:
		ind = -1
	else:
		ind = v[0]+v[1]+v[2]+v[3]-v[4]-v[5]-v[6]-v[7]
	return ind
	
def CombCont(I):
# Combinatorial contraction
	l = 0
	n = 0
	while l< 1:
		for i in arange(1,img.size[1]):
			for j in arange(1,img.size[0]):
				if I[i,j] == 1 and index(I,i,j) == 1:
					I[i,j] = 0
					n = n + 1
		if n == 0:
			l = 10
		else: 
			n = 0	
	return I	
	
def Euler(I):
	# Calculation of Euler characteristics
	l = 0
	n = 0
	k = 0
	I = CombCont(I)
	while l<1:
		for i in arange(1,img.size[1]):
			for j in arange(1,img.size[0]):
				if I[i,j] == 1 and index(I,i,j) == 2:
					I[i,j] = 0
					I = CombCont(I)
					k = k + 1
					n = n + 1			
		if k == 0:
			l = 10
			E = sum(I) - n
		else:
			k = 0
	return E	

def EulerVer1(I):
	# Calculation of Euler characteristic by definition
	V = zeros((img.size[1],img.size[0]))
	R = zeros((img.size[1],img.size[0],2))
	G = sum(I)
	for i in arange(0,img.size[1]):
			for j in arange(0,img.size[0]):
				if I[i,j] == 1:
					V[i,j] = 1
					R[i,j,0] = 1
					R[i,j,1] = 1
				else:	
					if I[i,j-1] == 1:
						V[i,j] = 1
						R[i,j,0] = 1
					if I[i-1,j-1] == 1:
						V[i,j] = 1
					if I[i-1,j] == 1:
						V[i,j] = 1
						R[i,j,1] = 1
	E = sum(V)-sum(R)+G
	return E
	
def EulerVer2(I):
	# Calculation of Euler characteristic by definition
	V = zeros((img.size[1],img.size[0]))
	R = zeros((img.size[1],img.size[0],2))
	G = sum(I)
	for i in arange(0,img.size[1]):
			for j in arange(0,img.size[0]):
				V[i,j] = amax([I[i,j],I[i,j-1],I[i-1,j-1],I[i-1,j]])
				R[i,j,0] = amax([I[i,j],I[i,j-1]])
				R[i,j,1] = amax([I[i,j],I[i-1,j]])
	B = sum(V)-sum(R)+G
	return B
"""	
def EulerVer3(I):
	E = 0
	# Calculation of Euler characteristic by definition
	for i in arange(1,shape(I)[1]):
		for j in arange(1,shape(I)[0]):
			V = I[i,j] or I[i,j-1] or I[i-1,j-1] or I[i-1,j]
			R1 = I[i,j] or I[i,j-1]
			R2 = I[i,j] or I[i-1,j]
			E += V + (-1)*R1 + (-1)*R2
	E += sum(I)			
	return E
"""		
def sliding_window(I, stepSize, windowSize):
	S = [[],[]]
	# slide a window across the image
	for y in xrange(0, img.size[0], stepSize):
		for x in xrange(0, img.size[1], stepSize):
			I1 = I[y:y + windowSize[1], x:x + windowSize[0]]
			E = Euler(I1)
			# Number of holes
			H = img.size[1]*img.size[0] - sum(I) - E
			#Number of connected components
			C = E + H
			S.append([[C],[H]])
			# yield the current window
	return (S)	
"""

E = []	
for i in arange(1,161):
	img = Image.open('D:\\Pattern recognition\\Small texture\\linsseeds1\\image{0:03d}.png'.format(i))
	img = img.convert('L')
	img = img.point(lambda x: 0 if x<128 else 255, '1')
	I = array(img)

	# zeropad the border
	for i in arange(0,img.size[1]):
		I[i,0] = 0
		I[i,img.size[1]-1] = 0
	for j in arange(0,img.size[0]):
		I[0,j] = 0
		I[img.size[0]-1,j] = 0
		
	#calculate Euler characteristics	
	x = EulerVer3(I)
	E.append(x)
	print x

E = sort(E)	
M = mean(E)
sigma = std(E)
fit = stats.norm.pdf(E, mean(E), std(E))  #fitting the distribution

print M, sigma

		
# plot histogram
fig = plt.figure()
plt.plot(E,fit,'-o')
plt.hist(E, bins =20, histtype='stepfilled', normed = True)
#plt.savefig('stone1.pdf', format='pdf')
plt.show()

"""
#read image
img0 = Image.open('binary1.png')#('D:\\Pattern recognition\\Texture\\sample42\\BW\\image001.png')
size = min(img0.size[0],img0.size[1])
img = img0.crop((0,0,size-1,size-1)) 
rgb = array(img)

#convert to mono
I = mono(rgb)
print(I[32,32])
#I = rgb[:,:]

(E,V,R,G) = EulerSimple(I)

print(E)
print(sum(V))
print(R)
print(G)


# Combinatorial contraction
I = CombCont(I)
showIm(I)


# Calculation of Euler characteristics
E = Euler(I)

# Number of holes
H = sum(I) - E
#Number of connected components
C = E + H

print( 'Euler characteristic: %s' ) %E
print( 'Connected components: %s' ) %C
print( 'Holes: %s' ) %H

S = sliding_window(I, 2, img.size[0]/10)

print(S)
"""
