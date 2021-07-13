#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIP assignment 1

1. Popularity Algorithm

"""
import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
from scipy.spatial import KDTree

def euclidean_dist(rgb, color):
    return math.sqrt((int(color[0])-rgb[0])**2 + (int(color[1])-rgb[1])**2 + 
                     (int(color[2])-rgb[2])**2)

def quantize(img, colormap):
     #quantized image
    
    
    #change colors with nearest representative color
    m,n,_ = img.shape
    colormap = np.array(colormap)
    tree = KDTree(colormap.copy())
    for i in range(m):
        for j in range(n):
            #replace colors
            d = tree.query(img[i,j])[1]
            img[i,j] = (tree.data)[d]
    

def popularity(img, colors, colormap):
    #histogram
    hist = {}
    
    #calculate 3d histogram
    m,n,_ = img.shape
    for i in range(m):
        for j in range(n):        
            
            rgb = tuple(img[i,j])
            
            hist[rgb] = hist.get(rgb, 0) + 1
    

    #sort histogram in decreasing order of frequency
    hist = dict(sorted(hist.items(), reverse=True, key=lambda i: i[1]))        
    colormap = list(hist.keys())[:colors]
    quantize(img, colormap.copy())
    return colormap
   


#read image as rgb
img = cv2.imread('/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
quant_img = img.copy()

#find colormap, give number of color as input to the function popularity
colormap = []
colormap = popularity(quant_img, 4, colormap)

#plot images
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(quant_img)


# write_img = cv2.cvtColor(quant_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('popularity_result.png', quant_img)


