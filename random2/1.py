from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import radon, rescale
from skimage.transform import iradon, rescale

# image = imread(data_dir + "/phantom.png", as_grey=True)
# image = rescale(image, scale=0.4)
img1 = imread('May1L.jpg',as_grey=True)  #queryimage # left image
img2 = imread('May1R.jpg',as_grey=True)

# plt.figure(figsize=(8, 4.5))

plt.subplot(221),plt.title("Left"),plt.imshow(img1)
plt.subplot(222),plt.title("Right"),plt.imshow(img2)

theta = np.linspace(0., 180., max(img1.shape), endpoint=True)

sinogramL = radon(img1, theta=theta, circle=False)
sinogramR = radon(img2, theta=theta, circle=False)

plt.subplot(223)
plt.title("Radon transform\n(Sinogram)")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Projection position (pixels)")
plt.imshow(sinogramL)

plt.subplot(224)
plt.title("Radon transform\n(Sinogram)")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Projection position (pixels)")
plt.imshow(sinogramR)

plt.show()

img3 = iradon(sinogramL)
img4 = iradon(sinogramR)

plt.subplot(121),plt.imshow(img3)
plt.subplot(122), plt.imshow(img4)
plt.show()
