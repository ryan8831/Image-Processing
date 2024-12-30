#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


c=255
y=float(input("y="))
count=0
for i in ['Cameraman.bmp','Peppers.bmp','Jetplane.bmp','Lake.bmp']:
    count+=1
    img = cv2.imread(i,0)
    H,W= img.shape[:]
    gamma=np.zeros((H,W),dtype='uint8')
    #s=cr^y
    for i in range(H):
        for j in range(W):
            gamma[i][j] = c*((img[i][j]/c)**y)
    result = cv2.hconcat([img,gamma])
    cv2.imshow('result',result)
    s='power-law'+str(count)+'.png'
    cv2.imwrite(s, result)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




