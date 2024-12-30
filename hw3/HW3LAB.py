#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2 
import numpy as np


#choose read which image
#img = cv2.imread('aloe.jpg')
#img = cv2.imread('church.jpg')
#img = cv2.imread('kitchen.jpg')
img = cv2.imread('house.jpg')
#set parameter
rgb2xyz = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])

xyz2rgb = np.array([[ 3.240479,-1.537150,-0.498535],
                    [-0.969256, 1.875992, 0.041556],
                    [ 0.055648,-0.204043, 1.057311]])

kernel= np.array([[-1.0,-1.0,-1.0],
                  [-1.0, 9.0,-1.0],
                  [-1.0,-1.0,-1.0]])

xyz_n = np.array([0.9515, 1.0000, 1.0886])
buffer = np.zeros( 3, np.float64)
f_xyz = np.zeros( 3, np.float64)

#print height and width of input image
H,W,C = img.shape[:]

#set parameter
normal_img = np.zeros( (H,W,C), np.float64 )
xyz_img = np.zeros( (H,W,C), np.float64 )
laplab_img = np.zeros( (H,W,C), np.float64 )
lab_img = np.zeros( (H,W,C), np.float64 )
liblab_img = np.zeros( (H,W,C), np.uint8 )
rgb_img = np.zeros( (H,W,C), np.float64 )

#normalization
normal_img[:,:,:] = img[:,:,:]/255

#RGB transform to XYZ space
for i in range(H):
    for j in range(W):
        xyz_img[i, j, :] = np.dot(rgb2xyz, normal_img[i,j,:])

#XYZ transform to L*A*B space
for i in range(H):
    for j in range(W):
        buffer[:] = xyz_img[i, j, :]/xyz_n[:]
        
        for k in range(3):
            if(buffer[k] > 0.008856):
                f_xyz[k] = math.pow(buffer[k], 1/3)
            else:
                f_xyz[k] = 7.787 * buffer[k] +16/116
        #L*
        if(buffer[1] > 0.008856):
            lab_img[i, j, 0] = 116 * math.pow(buffer[1], 1/3) - 16
        else:
            lab_img[i, j, 0] = 903.3 * buffer[1]
        
        #A*
        lab_img[i, j, 1] = 500 * (f_xyz[0] - f_xyz[1])
        
        #B*
        lab_img[i, j, 2] = 200 * (f_xyz[1] - f_xyz[2])
        
liblab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


laplab_img[:, :, :] = lab_img[:, :, :]

for i in range(H-2):
    for j in range(W-2):
        laplab_img[i+1, j+1, 0] = np.sum(kernel * lab_img[i:i+3, j:j+3, 0])
        if(laplab_img[i+1, j+1, 0] < 0):laplab_img[i+1, j+1, 0] = 0
        if(laplab_img[i+1, j+1, 0] >= 255):laplab_img[i+1, j+1, 0] = 255



#L*A*B transform to XYZ space
for i in range(H):
    for j in range(W):
        f_xyz[1] = (laplab_img[i, j, 0] + 16) / 116
        f_xyz[0] =  f_xyz[1] + laplab_img[i, j, 1]/500
        f_xyz[2] =  f_xyz[1] - laplab_img[i, j, 2]/200
        #X
        if(f_xyz[0] > 0.008856):
            xyz_img[i, j, 0] = xyz_n[0] * (f_xyz[0]**3)
        else:
            xyz_img[i, j, 0] = ((f_xyz[0]-16) / 116) * 3 * (0.008865**2) * xyz_n[0]
        #Y
        if(f_xyz[1] > 0.008856):
            xyz_img[i, j, 1] = xyz_n[1] * (f_xyz[1]**3)
        else:
            xyz_img[i, j, 1] = ((f_xyz[1]-16) / 116) * 3 * (0.008865**2) * xyz_n[1]
        #Z
        if(f_xyz[2] > 0.008856):
            xyz_img[i, j, 2] = xyz_n[2] * (f_xyz[2]**3)
        else:
            xyz_img[i, j, 2] = ((f_xyz[2]-16) / 116) * 3 * (0.008865**2) * xyz_n[2]
      
#XYZ transform to RGB space
for i in range(H):
    for j in range(W):
        rgb_img[i, j, :] = np.dot(xyz2rgb, xyz_img[i,j,:])

#non-ormalization
rgb_img[:,:,:] = rgb_img[:,:,:] * 255
rgb_img = np.maximum(rgb_img, 0)
rgb_img = np.minimum(rgb_img, 255)


cv2.imshow('source', img)
cv2.imshow('LAB', lab_img.astype(np.uint8))
cv2.imshow('LAP_LAB', laplab_img.astype(np.uint8))
cv2.imshow('lib_LAB', liblab_img)
cv2.imshow('RGB', rgb_img.astype(np.uint8))
cv2.imwrite('LAB_house.jpg', rgb_img.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




