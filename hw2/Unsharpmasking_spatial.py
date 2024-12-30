#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#img = cv2.imread('blurry_moon.tif',0)
img = cv2.imread('skeleton_orig.bmp',0)
H,W = img.shape[:]
Unsharp_Maskimg= np.zeros((H, W), np.uint8)
padimg=np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0) 
Laplacian_img=np.zeros((H, W), np.uint8)
k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])


# In[5]:


def meanfilter(x,y,img):
    result=0
    for i in range(0,3):
        for j in range(0,3):
            result=result+img[x+i][y+j]*k[i][j]
    result=result//9
    return result

for i in range(0,H-2):
    for j in range(0,W-2):
        temp=meanfilter(i,j,padimg)
        if(temp)<0 :
            temp=0
        if(temp)>255 :
            temp=255
        Laplacian_img[i][j]=temp  

Mask=(img/255)-Laplacian_img
for i in range(0,H):
    for j in range(0,W):
        temp=(img[i][j])-(Mask[i][j])
        if(temp)<0 :
            temp=0
        if abs(temp)>255 :
            temp=255
        Unsharp_Maskimg[i][j]=np.round(temp)
result = cv2.hconcat([img,Unsharp_Maskimg])
cv2.imshow('result',Mask)
cv2.waitKey(0)
cv2.imshow('result',result)
cv2.imwrite('skeleton_origUnsharpmasking_spatial.jpg', Unsharp_Maskimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




