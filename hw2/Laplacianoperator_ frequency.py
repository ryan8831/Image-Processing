#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft

img = cv2.imread('blurry_moon.tif',0)
#img = cv2.imread('skeleton_orig.bmp',0)
H,W = img.shape[:]
result_img = np.zeros((H, W), np.uint8)
fft2 = np.fft.fft2(img)
lap_fft2 = np.fft.fft2(img)

#Centralization
shift2center = np.fft.fftshift(fft2)
shift2center[int((H/2)-1) : int((H/2)+1), int((W/2)-1) : int((W/2)+1)] = 0

#laplacian sharpening
for i in range(H):
    for j in range(W):
        lap_fft2[i][j] = -4*(math.pi**2)*abs((i-H/2)**2 + (j-W/2)**2)*shift2center[i][j]

#Inverse Centralization
center2shift = np.fft.ifftshift(lap_fft2)

#Inverse Fourier transform
ifft2 = np.fft.ifft2(center2shift)


lap_img = np.abs(ifft2)/np.max(np.abs(ifft2))
result_img = lap_img + (img/255)

cv2.imshow('input image', img)
cv2.imshow('result', result_img)
result_img=result_img*255
cv2.imwrite('blurry_moon_Lap_frequency.jpg', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




