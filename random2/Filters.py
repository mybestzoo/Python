#filters
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fftshift, fft, ifft
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

order = 512
# zero pad input image
# construct the fourier filter
delta = 1
l1 = (2*np.pi)**(-4/5) * (delta)**(8/5) /5
l2 = (2*np.pi)**(-4/5) * (delta)**(-2/5) *4/5

print(l1)
print(l2)

freqs = np.zeros((order, 1))

f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
w = 2 * np.pi * f

x = np.linspace(0, 2*np.pi, 32)
x = x.reshape(-1,1)
z = l2 / (l1 * ((x)**5 / (2*np.pi)) + l2) + np.sqrt(l1*l2) * (x)**2 * np.sqrt(l1*((x)**5 / (2*np.pi)) + l2 - x/(2*np.pi)) / (l1 * ((x)**5 / (2*np.pi)) + l2)

#"tigran":
g = l2 / (l1 * ((w)**5 / (2*np.pi)) + l2) - np.sqrt(l1*l2) * (w)**2 * np.sqrt(l1*((w)**5 / (2*np.pi)) + l2 - w/(2*np.pi)) / (l1 * ((w)**5 / (2*np.pi)) + l2)
f[1:] = (l2 / (l1 * ((w[1:])**5 / (2*np.pi)) + l2) - np.sqrt(l1*l2) * (w[1:])**2 * np.sqrt(l1*((w[1:])**5 / (2*np.pi)) + l2 - w[1:]/(2*np.pi)) / (l1 * ((w[1:])**5 / (2*np.pi)) + l2))
f[w<(2*np.pi)**(1/5)] = (l2 / (l1 * ((w[w<(2*np.pi)**(1/5)])**5 / (2*np.pi)) + l2) + np.sqrt(l1*l2) * (w[w<(2*np.pi)**(1/5)])**2 * np.sqrt(l1*((w[w<(2*np.pi)**(1/5)])**5 / (2*np.pi)) + l2 - w[w<(2*np.pi)**(1/5)]/(2*np.pi)) / (l1 * ((w[w<(2*np.pi)**(1/5)])**5 / (2*np.pi)) + l2))
f[w < 2*np.pi*l2] = 1
#f[w > 2*np.pi*l2] = l2 / (l1 * ((w[1:])**5 / (2*np.pi)) + l2) - np.sqrt(l1*l2) * (w[1:])**2 * np.sqrt(l1*((w[1:])**5 / (2*np.pi)) + l2 - w[1:]/(2*np.pi)) / (l1 * ((w[1:])**5 / (2*np.pi)) + l2)
#f[w > l1**(-1/4)] = 0
#"filter":
d = np.sin(x / 2) / (x / 2)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(w,g,'r')
ax1.plot(x,z,'g')
ax1.plot(w,f,'b')
ax1.plot(x,d,'o')
plt.show()