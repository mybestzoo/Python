from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def toImage(I):
	img = Image.new('L',(shape(I)[0],shape(I)[1]))
	pixels = img.load()
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			pixels[i, j] = I[j,i]
	return img

def function(x):
	coeff = [0,0,-1/2,1/3,-1/4,1/5,-1/6,1/7,-1/8,1/9,-1/10,1/11,-1/12,1/13,-1/14,1/15,-1/16,1/17,-1/18,1/19,-1/20,1/21]
	# coefficients of the derivative
	dcoeff = zeros(len(coeff))
	for n in range(0,len(coeff)):
		dcoeff[n] = (-1)*coeff[n]*n
	# L2 norm of the derivative
	norm = linalg.norm(dcoeff)
	#normalized coefficients
	ncoeff = coeff / norm
	f = zeros(len(x)) 
	for i in range(0,len(x)):
		for n in range(0,len(coeff)):
			f[i] = f[i]+coeff[n]*cos(x[i]*n)
	return f

def Normalize(im,C):
	# Divide the function values by the norm of the derivative to fit the class: |f'|<=C
	# DCT coefficients
	norm = zeros(len(im))
	for i in range(0,len(im)):
		y = DCT(im[i,:],len(im))
		# Derivative coefficients
		d = zeros(len(y))
		for n in range(0,len(y)):
			d[n] = (-1)*y[n]*n
		# L2 norm of the derivative
		norm[i] = linalg.norm(d)
	im_normalized = C*im/max(norm)
	print(max(norm))
	return im_normalized
	
def C(N):
	# the constant  that specifies the class \|f'\|<=C
	# partial sum of sum_0^(N-1) i^2
	partialS = (N-1)*N*(2*N-1)
	C = 128*sqrt(partialS)
	return C

def DCT(f,N):
	# f - signal values
	# N - number of coefficients to compute
	# DCT Matrix
	Phi = zeros((N,len(f)))
	for j in range(0,len(f)):
		Phi[0,j] = 1/sqrt(len(f))
	for i in range(1,N):
		for j in range (0,len(f)):
			Phi[i,j] = sqrt(2/len(f)) * cos(i*(2*j+1)*pi/(2*len(f)))	
	# DCT coefficients of f
	DCT = dot(Phi,f)
	return DCT

def iDCT(f,N):
	# f - DCT coefficients
	# N - number of nodes
	# Inverse DCT matrix is Phi.T
	Phi = zeros((N,len(f)))
	for i in range(0,N):
		Phi[i,0] = 1/sqrt(len(f))
	for i in range(0,N):
		for j in range (1,len(f)):
			Phi[i,j] = sqrt(2/len(f)) * cos(j*(2*i+1)*pi/(2*len(f)))
	# inverse DCT of f
	iDCT = dot(Phi,f)
	return iDCT

def Opt(f,N,delta,C):
	# Define p
	S=0
	k=0
	while S<=C**2 and k<=len(f)-1:
		S = S + delta[k]**2*k**2
		k = k+1
	p = k-1
	
	print('Number of coefficients in optimal method:',p+1)
	
	# define filter a
	a = ones(N)
	if p < N-1:
		a = zeros(N)
		for i in range(0,p+1):
			a[i] = 1-(i**2)/((p+1)**2)
	#optimal recovery of f
	Opt = iDCT(f*a,N)
	return Opt	

def Quantization(y,quant):
	# quant - parameter of quantization
	# divide q by a constant
	q = y/quant
	# round q to the integer values
	for i in range(0,len(q)):
		if q[i] >= 0:
			q[i] = int(floor(q[i]))
		else:
			q[i] = int(ceil(q[i]))
	# multiply q by the same constant
	q = (q+random.rand(len(q)))*quant
	return q

C = 7200
N = 64
quant = 2*20
delta = ones(N)*quant#/2
	
img = Image.open('sasha.jpg')
I = img.convert('L')
I.save('origunnorm.png')

img = array(I)
Normalize(img,C)

naturalMat = zeros((N,N))
optimalMat = zeros((N,N))
originalMat = zeros((N,N))

for i in range(0,N):
	f = img[i,:]
	y = DCT(f,N)
	g = Quantization(y,quant)
	#g_opt = zeros(len(g))
	#for j in range(0,len(g)):
	#	if g[j]>0:
	#		g_opt[j] = g[j] + 1/2 * quant
	#	elif g[j] <0:
	#		g_opt[j] = g[j] - 1/2 * quant 
	#	else:
	#		if y[j] >=0:
	#			g_opt[j] = g[j] + 1/2 * quant
	#		else:
	#			g_opt[j] = g[j] - 1/2 * quant
	u = iDCT(g,N)
	v = Opt(g,N,delta,C)
	originalMat[i,:] = f
	naturalMat[i,:] = u
	optimalMat[i,:] = v

print ('Error of natural method:', linalg.norm(originalMat-naturalMat))
print ('Error of optimal method:', linalg.norm(originalMat-optimalMat))		
	
original = toImage(originalMat)
original.save('original.png')

natural = toImage(naturalMat)
natural.save('natural.png')

optimal = toImage(optimalMat)
optimal.save('optimal.png')
