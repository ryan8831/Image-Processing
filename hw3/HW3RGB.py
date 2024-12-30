#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
#img = cv2.imread('aloe.jpg')
#img = cv2.imread('church.jpg')
#img = cv2.imread('kitchen.jpg')
img = cv2.imread('house.jpg')
H,W,CH = img.shape[:]
lap = np.zeros( (H, W, CH), np.float64)
kernel = np.array([[-1,-1,-1],
                     [-1, 9,-1],
                     [-1,-1,-1]])

for i in range(H-2):
    for j in range(W-2):
        for k in range(CH):
            lap[i, j, k] = (np.sum(kernel * img[i:i+3, j:j+3, k]))
            if(lap[i, j, k] < 0):lap[i, j, k] = 0
            if(lap[i, j, k] >= 255):lap[i, j, k] = 255

cv2.imshow('input', img)
cv2.imshow('laplacian', lap.astype(np.uint8))
cv2.imwrite('RGB_house.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




