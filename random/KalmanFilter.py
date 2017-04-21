#http://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf

from numpy import * #dot, sum, tile, linalg
from numpy.random import randn
from numpy.linalg import inv, det
import numpy as np
#import matplotlib 
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
#import matplotlib.animation as manimation
from scipy.interpolate import interp1d


def kf_predict(X, P, A, Q, B, U):
	X = dot(A, X) + dot(B, U)
	P = dot(A, dot(P, A.T)) + Q
	return(X,P)

def kf_update(X, P, Y, H, R):
	IM = dot(H, X)
	IS = R + dot(H, dot(P, H.T))
	K = dot(P, dot(H.T, inv(IS)))
	X = X + dot(K, (Y-IM))
	P = P - dot(K, dot(IS, K.T))
	LH = gauss_pdf(Y, IM, IS)
	return (X,P,K,IM,IS,LH)

def gauss_pdf(X, M, S):
	if M.shape[1] == 1:
		DX = X - tile(M, X.shape[1])
		E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
		E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
		P = exp(-E)
	elif X.shape()[1] == 1:
		DX = tile(X, M.shape[1])- M
		E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
		E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
		P = exp(-E)
	else:
		DX = X-M
		E = 0.5 * dot(DX.T, dot(inv(S), DX))
		E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(det(S))
		P = exp(-E)
	return (P[0],E[0])
	
	
#time step of mobile movement
dt = 0.1

# Initialization of state matrices
X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])
Q = array([[pow(dt,4)/float(4), 0, pow(dt,3)/float(2) , 0], [0, pow(dt,4)/float(4), 0, pow(dt,3)/float(2)], [pow(dt,3)/float(2), 0, pow(dt,2), 0], [0, pow(dt,3)/float(2), 0,pow(dt,2)]])#eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))

# Measurement matrices
Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +abs(randn(1)[0])]])
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = array([[0.2, 0], [0, 0.2]])#eye(Y.shape[0])

# Number of iterations in Kalman Filter
N_iter =100

# Allocate space for arrays
StateX = zeros(N_iter)
StateY = zeros(N_iter)
EstimX = zeros(N_iter)
EstimY = zeros(N_iter)
MeasX = zeros(N_iter)
MeasY = zeros(N_iter)

# Create states
for i in range(0,N_iter):
	StateX[i] = i*5/float(N_iter)
	StateY[i] = 10*np.sin(1.5*StateX[i])
	
# Applying the Kalman Filter
for i in arange(0, N_iter):
	(X, P) = kf_predict(X, P, A, Q, B, U)
	Y = array([[StateX[i] + (0.01* randn(1)[0])],[StateY[i] +(0.01* randn(1)[0])]])
	(X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
	
	EstimX[i]=X[0]
	EstimY[i]=X[1]
	
	MeasX[i]=Y[0]
	MeasY[i]=Y[1]
	
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(MeasX, MeasY, s=10, c='g', marker="s", label='Measurement')
ax1.plot(StateX,StateY, '.r-', label='State')
ax1.plot(StateX,StateY, '.r-', label='State')
ax1.plot(EstimX,EstimY, 'b-', label='Estimation')
plt.legend(loc='upper left');
plt.show()
#plt.savefig('KF.pdf', format='pdf')
