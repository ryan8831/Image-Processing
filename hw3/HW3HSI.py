#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2 
import numpy as np
#img = cv2.imread('aloe.jpg')
#img = cv2.imread('church.jpg')
#img = cv2.imread('kitchen.jpg')
img = cv2.imread('house.jpg')
H,W,C = img.shape[:]
normal_img = np.zeros( (H, W, C), np.float64 )
hsi_img = np.zeros( (H, W, C), np.float64 )
laphsi_img = np.zeros( (H, W, C), np.float64 )
hsv_img = np.zeros( (H, W, C), np.uint8 )
rgb_img = np.zeros( (H, W, C), np.float64 )

kernel= np.array([[-1.0,-1.0,-1.0],
                  [-1.0, 9.0,-1.0],
                  [-1.0,-1.0,-1.0]])


normal_img[:,:,:] = img[:,:,:]/255

#RGB transform to HSI space
#H
for i in range(H):
    for j in range(W):
        sqrt = math.sqrt((normal_img[i,j,2] - normal_img[i,j,1])**2
                                +(normal_img[i,j,2] - normal_img[i,j,0])
                                *(normal_img[i,j,1] - normal_img[i,j,0]))
        if(sqrt != 0):
            theta = np.arccos(0.5*(normal_img[i,j,2] - normal_img[i,j,1]+ normal_img[i,j,2] - normal_img[i,j,0])/sqrt)    
            if(normal_img[i, j, 0] <= normal_img[i, j, 1]):
                hsi_img[i,j,0] = theta
            elif(normal_img[i, j, 0] > normal_img[i, j, 1]):
                hsi_img[i,j,0] = 2*np.pi - theta
        else: hsi_img[i,j,0] = 0
        hsi_img[i,j,0] = hsi_img[i,j,0]/(2*np.pi)
        
#S
        if((normal_img[i, j, 0] + normal_img[i, j, 1] + normal_img[i, j, 2]) == 0):
            hsi_img[i, j, 1] = 0
        else:
            hsi_img[i, j, 1] = 1 - (normal_img[i, j, :].min() * 3 / (normal_img[i, j, 0] + normal_img[i, j, 1] + normal_img[i, j, 2]) )

#I
hsi_img[:,:,2] = (normal_img[:,:,0] + normal_img[:,:,1] + normal_img[:,:,2])/3

#non-ormalization
hsi_img[:,:,:] = hsi_img[:,:,:]*255

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)




laphsi_img[:, :, :] = hsi_img[:, :, :]


for i in range(H-2):
    for j in range(W-2):
        for k in range(C-1):
            laphsi_img[i+1, j+1, k+1] = np.sum(kernel * hsi_img[i:i+3, j:j+3, k+1])
            if(laphsi_img[i+1, j+1, k+1] < 0):laphsi_img[i+1, j+1, k+1] = 0
            if(laphsi_img[i+1, j+1, k+1] >= 255):laphsi_img[i+1, j+1, k+1] = 255



normal_img[:,:,:] = laphsi_img[:,:,:]/255

#HSI transform to RGB space
for i in range(H):
    for j in range(W):
        if(normal_img[i,j,0]*2*math.pi >= 0 and normal_img[i,j,0]*2*math.pi < 2*math.pi/3):
            #B
            rgb_img[i,j,0] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #R
            rgb_img[i,j,2] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*np.cos(normal_img[i,j,0]*2*math.pi)/np.cos(math.pi/3 - normal_img[i,j,0]*2*math.pi)))      
            #G
            rgb_img[i,j,1] = 3 * normal_img[i,j,2] - (rgb_img[i,j,2] + rgb_img[i,j,0])

        if(normal_img[i,j,0]*2*math.pi >= 2*math.pi/3 and normal_img[i,j,0]*2*math.pi < 4*math.pi/3):
            #R
            rgb_img[i,j,2] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #G
            rgb_img[i,j,1] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*math.cos(normal_img[i,j,0]*2*math.pi-2*math.pi/3)/math.cos(math.pi - normal_img[i,j,0]*2*math.pi)))
            #B
            rgb_img[i,j,0] = 3 * normal_img[i,j,2] - (rgb_img[i,j,2] + rgb_img[i,j,1])
            
        if(normal_img[i,j,0]*2*math.pi >= 4*math.pi/3 and normal_img[i,j,0]*2*math.pi < 6*math.pi/3):
            #G
            rgb_img[i,j,1] = normal_img[i,j,2] * (1 - normal_img[i,j,1])
            #B
            rgb_img[i,j,0] = normal_img[i,j,2] * (1 + (normal_img[i,j,1]*math.cos(normal_img[i,j,0]*2*math.pi-4*math.pi/3)/math.cos(5*math.pi/3 - normal_img[i,j,0]*2*math.pi)))
            #R
            rgb_img[i,j,2] = 3 * normal_img[i,j,2] - (rgb_img[i,j,1] + rgb_img[i,j,0])

#non-ormalization
rgb_img[:,:,:] = rgb_img[:,:,:]*255
rgb_img = np.maximum(rgb_img, 0)
rgb_img = np.minimum(rgb_img, 255)



cv2.imshow('source', img)
cv2.imshow('HSI', hsi_img.astype(np.uint8))
cv2.imshow('LAP_HSI', laphsi_img.astype(np.uint8))
cv2.imshow('LIB_HSV', hsv_img)
cv2.imshow('RGB', rgb_img.astype(np.uint8))
cv2.imwrite('HSI_house.jpg', rgb_img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




