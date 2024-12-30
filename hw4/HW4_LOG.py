#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2 
img = cv2.imread('image3.jpg',0)
Height, width = img.shape[:]
LOG_img= np.zeros((Height, width, 1), np.uint8)
LOG_mask=[[0,0,-1,0,0]
         ,[0,-1,-2,-1,0]
         ,[-1,-2,16,-2,-1]
         ,[0,-1,-2,-1,0]
         ,[0,0,-1,0,0]]
def convolution(x,y,img):
    result=0
    for i in range(0,5):
        for j in range(0,5):
            result=result+img[x+i,y+j]*LOG_mask[i][j]
    return result

for i in range(0,Height-4):
    for j in range(0,width-4):
        temp=convolution(i,j,img)
        if(temp)<0 :
            temp=0
        if(temp)>255 :
            temp=255
        LOG_img[i,j]=temp
        
cv2.imshow('source', img)
cv2.imshow('LOG_img', LOG_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image3_LOG.jpg',LOG_img)


# In[ ]:




