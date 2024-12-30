#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
#img = cv2.imread('blurry_moon.tif',0)
img = cv2.imread('skeleton_orig.bmp',0)
H,W = img.shape[:]
highboost_filteringimg=np.zeros((H, W), np.uint8)
padimg=np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0) 
Laplacian_img=np.zeros((H, W), np.uint8)
k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])


# In[6]:


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
        if abs(temp)>255 :
            temp=255
        Laplacian_img[i][j]=temp  
Mask=(img/255)-Laplacian_img
A=3
for i in range(0,H):
    for j in range(0,W):
        temp=img[i][j]-(A*Mask[i][j])
        if(temp)<0 :
            temp=0
        if abs(temp)>255 :
            temp=255
        highboost_filteringimg[i][j]=np.round(temp)
result = cv2.hconcat([img,highboost_filteringimg])
cv2.imshow('result',Mask)
cv2.waitKey(0)
cv2.imshow('result',result)
cv2.imwrite('skeleton_orig_high-boost_filtering_saptial.jpg', highboost_filteringimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




