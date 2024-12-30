#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
#img = cv2.imread('Cameraman.bmp',0)
#img = cv2.imread('Peppers.bmp',0)
#img = cv2.imread('Jetplane.bmp',0)
img = cv2.imread('Lake.bmp',0)
H,W = img.shape[:]
img2= np.zeros((H-2,W-2), np.uint8)
k=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])


# In[5]:


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
        img2[i][j]=temp
    
img = cv2.resize(img, (H-2, W-2), interpolation=cv2.INTER_AREA)
result = cv2.hconcat([img,img2])
cv2.imwrite('Laplacian1.png', img2)
cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




