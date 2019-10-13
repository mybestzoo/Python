from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

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
	return f, ncoeff

def Normalize(f):
	# Divide the function values by the norm of the derivative to fit the class: |f'|<=1
	# DCT coefficients
	y = DCT(f,len(f))
	# Derivative coefficients
	d = zeros(len(y))
	for n in range(0,len(y)):
		d[n] = (-1)*y[n]*n
	# L2 norm of the derivative
	norm = linalg.norm(d)
	fn = f/norm
	return fn

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

def Opt(f,N,delta):
	# Define p
	S=0
	k=0
	while S<=1 and k<=len(f)-1:
		S = S + delta[k]**2*k**2
		k = k+1
	p = k-1

	print('Nuber of harmonics in optimal method:',p+1)
	
	# define filter a
	a = zeros(p+1)
	for i in range(0,p+1):
		a[i] = 1-i**2/(p+1)**2
	Phi = zeros((N,p+1))
	for i in range(0,N):
		Phi[i,0] = 1/sqrt(len(f))
	for i in range(0,N):
		for j in range (1,p+1):
			Phi[i,j] = a[j]*sqrt(2/len(f)) * cos(j*(2*i+1)*pi/(2*len(f)))
	# inverse DCT of f
	Opt = dot(Phi,f[0:p+1])
	return Opt	
	
	# define the matrix of the optimal method

def Quant(y):
	# n - parameter of quantization
	n = 1.05
	# divide q by a constant
	q = y/n
	# round q to the integer values
	for i in range(0,len(q)):
		if q[i] >= 0:
			q[i] = int(floor(q[i]))
		else:
			q[i] = int(ceil(q[i]))
	# multiply q by the same constant
	q = q*2**n
	return q


# number of nodes
N = 32

# create nodes
x = zeros(N)
for n in range(0,N):
	x[n] = pi*(n+0.5)/N

# Values of function f in N nodes
#[f,nc] = function(x) 
f = x**5-x**2+1

# Normalize function f
f = Normalize(f)

# DCT coefficients of f
y = DCT(f,N)

# Quantization
g = Quant(y)
# calculate the last index of non zero element
i = len(g)-1
while g[i] == 0 and i > 0:
	i = i-1
print ('Number of harmonics after quantization:',i+1)

# Define the error delta THIS IS WRONG
delta = abs(g-y[0:len(g)])
print ('Max error:',max(delta))

# Reconstruct the signal by inverse DCT
u = iDCT(g,N)

# Reconstruct the signal by the optimal method
v = Opt(g,N,delta)

# plot graphs
plt.scatter(x, f, c ='g', label = 'Original')
plt.scatter(x, u, c = 'b', label = 'DCT')
plt.scatter(x, v, s=10, c='r', marker="s", label='Optimal')
plt.show()