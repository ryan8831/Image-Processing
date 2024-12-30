#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('Cameraman.bmp',0)
#img = cv2.imread('Peppers.bmp',0)
#img = cv2.imread('Jetplane.bmp',0)
img = cv2.imread('Lake.bmp',0)
H,W = img.shape[:]
pdf= np.zeros(256)
cdf= np.zeros(256)


# In[2]:


for i in range(H):
    for j in range(W):
        pdf[img[i][j]]+=1
s=H*W
for i in range(len(pdf)):
    cdf[i]=pdf[i]/s
for i in range(1,256):
    cdf[i]=cdf[i]+cdf[i-1]
    
img_hist=img.copy()
m=np.min(cdf)
M=np.max(cdf)
for i in range(H):
    for j in range(W):
        img_hist[i][j]=round((cdf[img[i][j]]-m)/(M-m)*255)
     
plt.subplot(211)
plt.hist(img.ravel(),256,[0,255])
plt.subplot(212)
plt.hist(img_hist.ravel(),256,[0,255])
plt.savefig('histogram4.png')
plt.show()
result = cv2.hconcat([img,img_hist])
cv2.imshow("result",result)
cv2.imwrite('his4.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




