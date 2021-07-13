#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIP assignment 1

3. Dithering

"""
import cv2
import math
from matplotlib import pyplot as plt
from part1a import popularity
from part1b import median_cut
from scipy.spatial import KDTree

def euclidean_dist(rgb, color):
    return math.sqrt((int(color[0])-rgb[0])**2 + (int(color[1])-rgb[1])**2 + 
                     (int(color[2])-rgb[2])**2)


def dithering(quant_img, dither_img):
    m,n,_ = img.shape
    colormap = []
        
    #popularity colormap
    #colormap = popularity(quant_img, 256, colormap)
    
    #uncomment for median cut colormap
    #Input color should be the power of 2 i.e. if you want to quantize the image
    # with 4 color then input 2 as 2^2=4 
    color = 2
    median_cut(colormap, quant_img, color)
    
    #Kd tree colormap
    tree = KDTree(colormap.copy())
    
    for i in range(m):
        for j in range(n):        
            #current value at (i,j)
            old_value = (dither_img[i,j])
         
            #find nearest color in kd tree
            d = tree.query(img[i,j], p = 2)[1]
            new_value  = (tree.data)[d]
            
            
            dither_img[i,j] = new_value
            e = old_value - new_value
            
            if(j < n-1):
                dither_img[i,j+1] = dither_img[i, j+1] + e*(5/16)
            if(i < m-1):
                dither_img[i+1,j] = dither_img[i+1, j] + e*(7/16)
            if(j < n-1 and i < m-1):
                dither_img[i+1,j+1] = dither_img[i+1, j+1] + e*(1/16)
            if(i > 0 and j < n-1):
                dither_img[i-1,j+1] = dither_img[i-1, j+1] + e*(3/16)
                

path = '/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/2.png'

#read image as rgb
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

quant_img=img.copy()

#dithering
dither_img = img.copy()
dithering(quant_img, dither_img)


#plot images
fig = plt.figure()
ax2 = fig.add_subplot(1,3,1)
ax2.imshow(img)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(quant_img)
ax2 = fig.add_subplot(1,3,3)
ax2.imshow(dither_img)

# write_img = cv2.cvtColor(dither_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('Dither_result.png', write_img)