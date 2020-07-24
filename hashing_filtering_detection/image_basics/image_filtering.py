import numpy as np
from matplotlib import pyplot as plt 

import cv2
from PIL import Image
from skimage.util import random_noise

img = cv2.imread('car1.png')
img_noise = random_noise(img, mode='s&p', amount=0.3)
plt.imsave('car2.png', img_noise)

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
fig.subplots_adjust(hspace=0.5)

img_new = cv2.imread('car2.png')
img_mean = cv2.blur(img_new, (3,3))
img_median = cv2.medianBlur(img_new, 3)
img_gaussian = cv2.GaussianBlur(img_new, (3,3), 0)

plt.subplot(221)
plt.imshow(img_new)
plt.title('base image')
plt.subplot(222)
plt.imshow(img_mean)
plt.title('mean filter image')
plt.subplot(223)
plt.imshow(img_median)
plt.title('median filter image')
plt.subplot(224)
plt.imshow(img_gaussian)
plt.title('gaussian filter image')