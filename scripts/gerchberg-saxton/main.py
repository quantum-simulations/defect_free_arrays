import cv2
from gs import *
import matplotlib.pyplot as plt
import numpy as np

filename = 'lattice.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
img = np.asarray(img, float)
max_iters = 20
phase_mask = Ger_Sax_algo(img, max_iters)
plt.figure(1)

plt.subplot(131)
im1 = plt.imshow(img)
plt.title('Desired image')
plt.colorbar(im1, fraction=0.046, pad=0.04)

plt.subplot(132)
im2 = plt.imshow(phase_mask)
plt.title('Phase mask')
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.subplot(133)
recovery = np.fft.ifft2(np.exp(phase_mask * 1j))
im3 = plt.imshow(np.absolute(recovery)**2)
plt.title('Recovered image')
plt.colorbar(im3, fraction=0.046, pad=0.04)

plt.show()
