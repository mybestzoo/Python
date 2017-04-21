from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#import photos

def MatToImage(I):
	# converts matrix into image
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

def NormD(f):
	# Norm of the derivative
	normD = zeros(len(f))
	y = DCT(f,len(f))
	# Derivative coefficients
	d = zeros(len(y))
	for n in range(0,len(y)):
		d[n] = (-1)*y[n]*n
	# L2 norm of the derivative
	normD = linalg.norm(d)
	return normD

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

def Opt(f,N,M,delta,C):
	# Define p
	S=0
	k=0
	while S<=C**2 and k<=len(f)-1:
		S = S + delta[k]**2*k**2
		k = k+1
	p = k-1
	
	#print('Number of coefficients in optimal method:',p+1)
	
	# define filter a
	a = ones(N)
	if p < N-1:
		a = zeros(N)
		for i in range(0,p+1):
			a[i] = 1-(i**2)/((p+1)**2)
			
	#optimal recovery of f
	Opt = iDCT(f*a,M)
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
	#q = (q+random.rand(len(q)))*quant
	q = q*quant
	return q


def CompressBlock(img,C,N,M,quant):
	
	delta = ones(N*15)*quant/2
	
	naturalMat = zeros((N,M))
	optimalMat = zeros((N,M))
	
	for i in range(0,N):
		f = img[i,:]
		C = NormD(f)
		y = DCT(f,N*15)
		g = Quantization(y,quant)
		# shift by 0.5*quant
		for j in range(0,len(g)):
			if g[j]>0:
				g[j] = g[j] + 0.5*quant
			if g[j]<0:
				g[j] = g[j] - 0.5*quant
		u = iDCT(g,M)
		v = Opt(g,N*15,M,delta,C)
		naturalMat[i,:] = u
		optimalMat[i,:] = v
	return naturalMat, optimalMat

	
# MAIN CODE 

C = 1
N = 4
M = 2*N
quant = 2**0

img = Image.open('sashaBig.png')
#img = photos.pick_image(True)
I = img.convert('L')
Img = array(I)

#Downsample x2
ImgDownSample = zeros((len(Img)/2,len(Img)/2))
for i in range(0,len(ImgDownSample)):
	for j in range(0,len(ImgDownSample)):
		ImgDownSample[i,j] = Img[i,2*j]

natural = zeros((len(Img),len(Img)))
optimal = zeros((len(Img),len(Img)))

for i in range(0,64):
	print i
	for j in range(0,64):
		img = ImgDownSample[i*N:i*N+N,j*N:j*N+N]
		naturalMat, optimalMat = CompressBlock(img,C,N,M,quant)
		#original[i*N:i*N+N,j*N:j*N+N] = Img[i*N:i*N+N,j*N:j*N+N]
		natural[i*N:i*N+N,j*M:j*M+M] = naturalMat
		optimal[i*N:i*N+N,j*M:j*M+M] = optimalMat

print ('Error of natural method:', 
linalg.norm(Img[0:len(Img)/2,:]-natural[0:len(Img)/2,:]))
print ('Error of optimal method:', 
linalg.norm(Img[0:len(Img)/2,:]-optimal[0:len(Img)/2,:]))		
	
#original = MatToImage(original)
#photos.save_image(original)
#original.save('original.png')

natural = MatToImage(natural)
#photos.save_image(natural)
natural.save('natural.png')

optimal = MatToImage(optimal)
#photos.save_image(optimal)
optimal.save('optimal.png')

ImgDownSample = MatToImage(ImgDownSample)
#photos.save_image(natural)
ImgDownSample.save('downsample.png')
