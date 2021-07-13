#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIP assignment 1

2. Median Cut Algorithm

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


#flatten the image
def flatten_array(quant_img):
    quant_img_arr = []
    m,n,_ = quant_img.shape
    for i in range(m):
        for j in range(n):
            quant_img_arr.append([i, j, quant_img[i,j,0],quant_img[i,j,1],
                             quant_img[i,j,2]]) 
    return np.array(quant_img_arr)

        
#find mean of each color channel
def get_ColorPalette(colormap, quant_img, quant_img_arr):
    mean_ = np.mean(quant_img_arr, axis = 0)
    r_mean = mean_[2]
    g_mean = mean_[3]
    b_mean = mean_[4]
    
    colormap.append((r_mean,g_mean,b_mean))
    
    for i in quant_img_arr:
        x = i[0]
        y = i[1]
        quant_img[x][y][0] = r_mean
        quant_img[x][y][1] = g_mean
        quant_img[x][y][2] = b_mean
    


def recursive_split(colormap, quant_img, quant_img_arr, color):
    
    if quant_img_arr.size == 0:
        return
    
    #find average of current bucket
    if color == 0:
        return get_ColorPalette(colormap, quant_img, quant_img_arr)
        
    
    #get channel with max range and sort according to highest range
    range_list = []
    
    for i in range(3):
        range_list.append(np.max(quant_img_arr[:, i+2])-np.min(quant_img_arr[:, i+2]))
    
    sort_index = range_list.index(max(range_list))  #index of highest range
    quant_img_arr = quant_img_arr[np.argsort(quant_img_arr[:,sort_index+2])]
  
    l = len(quant_img_arr)
    median = (l+1)//2
    
    #split into two parts at the median
    recursive_split(colormap, quant_img, quant_img_arr[0:median], color-1)
    recursive_split(colormap, quant_img, quant_img_arr[median:], color-1)

    
def median_cut(colormap, quant_img, color):
    #flatten array
    quant_img_arr = flatten_array(quant_img)
    recursive_split(colormap, quant_img, quant_img_arr, color)
    
    
    
path = '/Users/thakore/Desktop/My Documents/Digital Image Analysis/Assignments/A1/Final Submission/2.png'
#read image as rgb
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
quant_img=img.copy()
colormap = []
#Input color should be the power of 2 i.e. if you want to quantize the image
# with 4 color then input 2 as 2^2=4 
color = 2
median_cut(colormap, quant_img, color)


#plot images
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(quant_img)


# write_img = cv2.cvtColor(quant_img, cv2.COLOR_RGB2BGR)
# cv2.imwrite("quant_Result.png", write_img) 

