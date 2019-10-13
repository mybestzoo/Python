from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale


image = imread(data_dir + "/phantom.png", as_grey=True)
image = rescale(image, scale=0.4)

#plt.figure(figsize=(8, 4.5))

#plt.subplot(121)
#plt.title("Original")
#plt.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=True)
sinogram = radon(image, theta=theta, circle=True)

fs = np.fft.fft(sinogram,None,0)

#plt.subplot(122)
#plt.title("Radon transform\n(Sinogram)")
#plt.xlabel("Projection angle (deg)")
#plt.ylabel("Projection position (pixels)")
#plt.imshow(sinogram, cmap=plt.cm.Greys_r,
#           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

#plt.subplots_adjust(hspace=0.4, wspace=0.5)
#plt.show()