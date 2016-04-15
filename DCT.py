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

def DCT(f):
	# DCT Matrix
	Phi = zeros((len(f),len(f)))
	for j in range(0,len(f)):
		Phi[0,j] = 1/sqrt(len(f))
	for i in range(1,len(f)):
		for j in range (0,len(f)):
			Phi[i,j] = sqrt(2/len(f)) * cos(i*(2*j+1)*pi/(2*len(f)))	
	# DCT coefficients of f
	DCT = dot(Phi,f)
	return DCT
	
def iDCT(f):
	# Inverse DCT matrix is Phi.T
	Phi = zeros((len(f),len(f)))
	for i in range(0,len(f)):
		Phi[i,0] = 1/sqrt(len(f))
	for i in range(0,len(f)):
		for j in range (1,len(f)):
			Phi[i,j] = sqrt(2/len(f)) * cos(j*(2*i+1)*pi/(2*len(f)))
	# inverse DCT of f
	iDCT = dot(Phi,f)
	return iDCT
	
def OptimalMethod(f,delta):
	# Define p
	S=0
	k=0
	while S<=1 and k<=len(f)-1:
		S = S + delta[k]**2*k**2
		k = k+1
	p = k-1
	
	# define filter a
	a = zeros(p+1)
	for i in range(0,p+1)
		a = 1-k**2/(p+1)^2 
	
	# define the matrix of the optimal method
	
		
# number of nodes
N = 15

# create nodes
x = zeros(N)
for n in range(0,N):
	x[n] = pi*(n+0.5)/N

# N values of function f in nodes
[f,nc] = function(x) 

y = DCT(f)

# Add error delta to DCT2 coefficients
delta = 0.00000001
g = y + delta*2*(random.random_sample(N)-1)

# Define the error delta
delta = abs(g-y)


"""
# DCT2
DCT2 = zeros(N)
for k in range(0,N):
	for n in range(0,N):
		DCT2[k] = DCT2[k] + y[n]*cos(pi*k*(2*n+1)/(2*N))
DCT2 = DCT2*2/N	
		


# DCT3
DCT3 = zeros(N)+DCT2[0]/2
for k in range(0,N):
	for n in range(1,N):
		DCT3[k] = DCT3[k]+DCT2[n]*cos(pi*(k+0.5)*n/N)
		
# Construct methods
resolution = 1000
t = linspace(0.,pi,resolution)

# natural method
E = zeros(resolution)+DCT2[0]/2
for i in range(0,resolution):
	for n in range(1,N):
		E[i] = E[i]+DCT2[n]*cos(t[i]*n)
		
# define p0
#min = min(N,len(nc))
#nc = nc[0:min]
#dct2 = DCT2[0:min]
#delta = max(dct2-nc)
delta = 0.1
print delta
S=0
k=0
while S<1:
	S = S + 2*delta**2*k**2
	k = k+1
p0 = k-1
if p0 > N:
	p0 = N
print(p0)

# optimal method
m = zeros(resolution)+DCT2[0]/2
for i in range(0,resolution):
	for n in range(1,p0):	
		m[i] = m[i]+DCT2[n]*(1-n**2/(p0+1)**2)*cos(t[i]*n)

# plot graphs
[f,nc] = function(t)
plt.plot(t, f, c ='g')
plt.plot(t, E, c = 'b')
plt.scatter(x, y, s=10, c='r', marker="s", label='Nodes')
plt.plot(t, m, c = 'r')
plt.show()

"""
