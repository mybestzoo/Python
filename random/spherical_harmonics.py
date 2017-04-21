# http://balbuceosastropy.blogspot.kr/2015/06/spherical-harmonics-in-python.html

%matplotlib inline
from __future__ import division
import scipy as sci
import scipy.special as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import math

def d_l(l):
	d = l*(l+1)
	return d
	
def m_l(l):
	if (l % 2 == 0):
		m = (-1)**(l/2)*2*np.sqrt(np.pi)*math.gamma((l+1)/2)/math.gamma((l+2)/2)
	else: 
		m = 0
	return m

d = [d_l(0),d_l(2),d_l(2),d_l(2),d_l(4),d_l(4),d_l(4),d_l(4),d_l(4)]
m = [m_l(0),m_l(2),m_l(2),m_l(2),m_l(4),m_l(4),m_l(4),m_l(4),m_l(4)]

coeff_true = [1,1,1,1,1,1,1,1,1]

# normalize coefficients
normalization_const = np.dot(np.multiply(d,d),np.multiply(coeff,coeff))
coeff_true = coeff_true/normalization_const
M_coeff = np.multiply(m,coeff_true)

# delta error
delta = np.random.rand(9)

M_coeff_noisy = M_coeff + (2*np.random.randint(2,size=9)-1)*delta

def function(coeff, PHI, THETA):
    f = coeff[0]*sp.sph_harm(0, 0, PHI, THETA).real
    for i in range(0,3):
        f = f + coeff[i+1]*sp.sph_harm(i, 2, PHI, THETA).real
    for i in range(0,5):
        f = f + coeff[i+4]*sp.sph_harm(i, 4, PHI, THETA).real
    return f
	
def define_p(delta):
	sum = 0
	for i in range(0,3):
		for k in range(0,2*i+1):
			sum = sum + ((d_l(2*i))**2)*delta[i*i+k]/(m_l(2*i))**2
		if sum <=1 :
			p = i
	return p

def lambda_1(p):
    lam = 1/(d_l(2*(p+1)))**2
    return lam

def lambda_l(l,p):
    yl = 1/(m_l(2*l))**2
    xl = (d_l(2*l))**2/(m_l(2*l))**2
    lambda_l = yl-lambda_1(p)*xl
    return lambda_l
	
def a_filter(p,l):
	if l>p:
		a_filter = 0
	else:
		lambda_1 = 1/(d_l(2*(p+1)))**2
		yl = 1/(m_l(2*l))**2
		xl = (d_l(2*l))**2/(m_l(2*l))**2
		lambda_l = yl-lambda_1*xl
		a_filter = lambda_l/(lambda_1*xl+lambda_l)
	return a_filter
	
def visualize(coeff):
	# Visualize real part of Y_4^2 as a heat map on the sphere
	# Coordinate arrays for the graphical representation
	x = np.linspace(-np.pi, np.pi, 100)
	y = np.linspace(-np.pi/2, np.pi/2, 50)
	X, Y = np.meshgrid(x, y)

	# Spherical coordinate arrays derived from x, y
	# Necessary conversions to get Mollweide right
	phi = x.copy()    # physical copy
	phi[x < 0] = 2 * np.pi + x[x<0]
	theta = np.pi/2 - y
	PHI, THETA = np.meshgrid(phi, theta)

	SH_SP = function(coeff,PHI,THETA) #sp.sph_harm(m, l, PHI, THETA).real    # Plot just the real part

	#This is to enable bold Latex symbols in the matplotlib titles, according to:
	#http://stackoverflow.com/questions/14324477/bold-font-weight-for-latex-axes-label-in-matplotlib
	matplotlib.rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

	xlabels = ['$210^\circ$', '$240^\circ$','$270^\circ$','$300^\circ$','$330^\circ$',
			'$0^\circ$', '$30^\circ$', '$60^\circ$', '$90^\circ$','$120^\circ$', '$150^\circ$']

	ylabels = ['$165^\circ$', '$150^\circ$', '$135^\circ$', '$120^\circ$', 
			'$105^\circ$', '$90^\circ$', '$75^\circ$', '$60^\circ$',
			'$45^\circ$','$30^\circ$','$15^\circ$']

	fig, ax = plt.subplots(subplot_kw=dict(projection='mollweide'), figsize=(10,8))
	im = ax.pcolormesh(X, Y , SH_SP)
	ax.set_xticklabels(xlabels, fontsize=14)
	ax.set_yticklabels(ylabels, fontsize=14)
	ax.set_title('real$(f)$', fontsize=20)
	ax.set_xlabel(r'$\boldsymbol \phi$', fontsize=20)
	ax.set_ylabel(r'$\boldsymbol{\theta}$', fontsize=20)
	ax.grid()
	fig.colorbar(im, orientation='horizontal');
	
visualize(coeff_true)

visualize(np.divide(M_coeff_noisy,m))]
print 'Error', np.linalg.norm(coeff_true - np.divide(M_coeff_noisy,m))

p = define_p(delta)
print 'p =',p
a = [a_filter(p,1),a_filter(p,1),a_filter(p,1),a_filter(p,1),a_filter(p,2),a_filter(p,2),a_filter(p,2),a_filter(p,2),a_filter(p,2)]
visualize(np.multiply(a,np.divide(M_coeff_noisy,m)))
print 'Error', np.linalg.norm(coeff_true - np.multiply(a,np.divide(M_coeff_noisy,m)))
