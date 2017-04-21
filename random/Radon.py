#Radon transform
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.fftpack import fftshift, fft, ifft, fftfreq
from skimage.io import imread
from skimage import data_dir, util
from skimage.transform import radon, rescale
from skimage.transform.radon_transform import _sinogram_circle_to_square
from numpy.random import randn
from scipy.stats import multivariate_normal

def iradonT(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=False):
    """
    Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional (default 'linear')
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")
    interpolation_types = ('linear', 'nearest', 'cubic')
    if not interpolation in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0])**2
                                               / 2.0)))
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)

    th = (np.pi / 180.0) * theta
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2**np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = util.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    #delta = 0.005
    l1 = (2*np.pi)**(-4/5) * (delta)**(8/5) /5
    l2 = (2*np.pi)**(-4/5) * (delta)**(-2/5) *4/5

    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter
    if filter == "ramp":
        pass
    elif filter == "tigran":
        g = fftfreq(projection_size_padded).reshape(-1, 1)
        w = abs(omega)
        g[1:] = l2 / (l1 * ((w[1:])**5 / (2*np.pi)) + l2) - np.sqrt(l1*l2) * (w[1:])**2 * np.sqrt(l1*((w[1:])**5 / (2*np.pi)) + l2 - w[1:]/(2*np.pi)) / (l1 * ((w[1:])**5 / (2*np.pi)) + l2)
        fourier_filter[1:] = fourier_filter[1:] * g[1:]
    elif filter == "shepp-logan":
        # Start from first element to avoid divide by zero
        fourier_filter[1:] = fourier_filter[1:] * np.sin(omega[1:]) / omega[1:]
    elif filter == "cosine":
        fourier_filter *= np.cos(omega)
    elif filter == "hamming":
        fourier_filter *= (0.54 + 0.46 * np.cos(omega / 2))
    elif filter == "hann":
        fourier_filter *= (1 + np.cos(omega / 2)) / 2
    elif filter is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)
    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    mid_index = radon_image.shape[0] // 2

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    # Reconstruct image by interpolation
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_filtered[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected
    if circle:
        radius = output_size // 2
        reconstruction_circle = (xpr**2 + ypr**2) <= radius**2
        reconstructed[~reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * len(th))

#MAIN PROGRAM
image = imread(data_dir + "/phantom.png", as_grey=True)

# smooth image with Gaussian filter
image = scipy.ndimage.filters.gaussian_filter(image,3)

fig = plt.figure()
plt.imshow(image, cmap=plt.cm.Greys_r)
plt.show()

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
#ax1.set_title("Original")
#ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)

#add noise
#x=np.arange(-2*np.pi,2*np.pi,4*np.pi/400)
#y=np.arange(-2*np.pi,2*np.pi,4*np.pi/400)
#xx, yy = np.meshgrid(x, y, indexing='ij')
#
#r = np.sqrt(xx**2 + yy**2)
#phi = np.arccos(x/r)
#
#z1 = np.sin(2*(r-np.pi)) / (r-np.pi)
#z2 = z1 * (1 + 0.5*np.cos(4*phi))
#sinogram = sinogram + 5*randn(400,400) #np.multiply(sinogram,1+0.2*z1)

#make error function
#step = np.shape(sinogram)[0]
#u = np.arange(0.0, 180.0, 180/step)
#v = np.arange(-1.0, 1.0, 2/step)
#U, V = np.meshgrid(u, v)
#a = randn(1,1)
#b = randn(1,1)
#ErNorm = np.sqrt( 2*np.pi * (b**2*(2+np.sin(2)) - a**2*(np.sin(2)-2)) / 2 )
#sinoErr = a*np.sin(V)+b*np.cos(V) / ErNorm

#add noise
NoiseLvl =  0.01  
delta = NoiseLvl*np.max(sinogram)
std = delta/np.sqrt(4*np.pi)
sinogram = sinogram + std*randn(400,400)

#reconstruct
reconstruction = iradonT(sinogram, theta=theta, filter = 'hann', circle = True)
reconstructionT = iradonT(sinogram, theta=theta, filter = 'tigran',circle = True)

error = np.abs(reconstruction - image)
#print('Natural reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))
print('Natural reconstruction error: %.3g' % np.linalg.norm(error))
errorT = np.abs(reconstructionT - image)
#print('Optimal reconstruction error: %.3g' % np.sqrt(np.mean(errorT**2)))
print('Optimal reconstruction error: %.3g' % np.linalg.norm(errorT))

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
ax1.set_title("Natural filter" )
ax1.imshow(reconstruction, cmap=plt.cm.Greys_r)
ax2.set_title("Optimal filter")
ax2.imshow(reconstructionT, cmap=plt.cm.Greys_r)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                              sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
ax1.set_title("Error of Natural filter")
ax1.imshow(error, cmap=plt.cm.Greys_r)
ax2.set_title("Error of Optimal filter")
ax2.imshow(errorT, cmap=plt.cm.Greys_r)
plt.show()

x = np.arange(0, 400, 1)
y = np.arange(0, 400, 1)
X, Y = np.meshgrid(x, y)

#plot reconstructed images
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Reconstruction by the 'natural' method")
ax1.contour(X,Y,reconstruction)
ax2.set_title("Reconstruction by the optimal method")
ax2.contour(X,Y,reconstructionT)
plt.show()

#plot slices
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,image[:,200], 'r',label='Gaussian')
ax1.plot(x,reconstruction[:,200], 'g',label='"Natural" method')
ax1.plot(x,reconstructionT[:,200], 'b',label='Optimal method')
plt.legend(loc='upper left');
plt.show()

#plot errors
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x,error[:,200], 'g',label='"Natural" method')
ax1.plot(x,errorT[:,200], 'b',label='Optimal method')
plt.legend(loc='upper left');
plt.show()
