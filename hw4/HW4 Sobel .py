#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import cv2 
img = cv2.imread('image3.jpg',0)
Height, width = img.shape[:]
Gaussian_img= np.zeros((Height-2, width-2, 1), np.uint16)
Gaussian_Filter=np.zeros((3, 3, 1),np.float16)
GX=np.zeros((Height-2, width-2, 1), np.int32)
GY=np.zeros((Height-2, width-2, 1), np.int32)
G=np.zeros((Height-2, width-2, 1), np.int32)

sigma =0.5**0.5
y, x = np.mgrid[-1:2, -1:2]
total=0
for i in range(3):
    for j in range(3):
        Gaussian_Filter[i][j]=(np.exp(-(x[i][j]**2+y[i][j]**2)/(2*sigma**2)))/(2*np.pi*sigma**2)
        total=total+Gaussian_Filter[i][j]
for i in range(3):
    for j in range(3):
        Gaussian_Filter[i][j]=np.round((Gaussian_Filter[i][j]/total),3)



def convolution(x,y,img):
    result=0
    for i in range(0,3):
        for j in range(0,3):
            result=result+img[x+i,y+j]*Gaussian_Filter[i][j]
    return float(result)

for i in range(0,Height-4):
    for j in range(0,width-4):
        temp=convolution(i,j,img)
        if(temp)<0 :
            temp=0
        if(temp)>255 :
            temp=255
        Gaussian_img[i,j]=np.round(temp,0)
cv2.imwrite('Gaussian.jpg', Gaussian_img)



# sobel detector 
Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#Gradient calculation
def dot(x,y,img,direction):
    result=0
    for i in range(0,3):
        for j in range(0,3):
            if(direction=='x'):
                result=result+img[x+i,y+j,0]*Gx[i][j]
            elif((direction=='y')):
                result=result+img[x+i,y+j,0]*Gy[i][j]
    return result

for i in range(0,Height-4):
    for j in range(0,width-4):
        temp1=dot(i,j,Gaussian_img,'x')
        temp2=dot(i,j,Gaussian_img,'y')
        GX[i][j]=temp1
        GY[i][j]=temp2
for i in range(0,Height-2):
    for j in range(0,width-2):
        G[i][j]=np.sqrt(((GX[i][j]**2)+(GY[i][j]**2)))


cv2.imshow('source', img)
cv2.imshow('sobel_img', G.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image3_sobel.jpg',G.astype(np.uint8))


# In[ ]:




