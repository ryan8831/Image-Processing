#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2 
import numpy as np


# In[10]:


img = cv2.imread('blurry_moon.tif',0)
#img = cv2.imread('skeleton_orig.bmp',0)
H,W = img.shape[:]
img2= np.zeros((H, W), np.uint8)
k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

def convolution(x,y,img):
    result=0
    for i in range(0,3):
        for j in range(0,3):
            result=result+img[x+i,y+j]*k[i][j]
    return result

for i in range(0,H-2):
    for j in range(0,W-2):
        temp=convolution(i,j,img)
        if(temp)<0 :
            temp=0
        if(temp)>255 :
            temp=255
        img2[i,j]=temp
cv2.imwrite('skeleton_orig_Lap_spatial.jpg', img2)
cv2.imshow('result',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




