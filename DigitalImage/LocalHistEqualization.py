#!/usr/bin/env python3
#-*- Skimage Gallery -- Local Histogram Equalization -*-
# 局部直方图均衡化
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

matplotlib.rcParams['font.size'] = 9

def plot_img_and_hist(image, axes, bins=256):
  '''Plot 一张图和它的直方图,累积直方图'''
  ax_img, ax_hist = axes
  ax_cdf = ax_hist.twinx()
  
  # Display image
  ax_img.imshow(image, cmap=plt.cm.gray)
  ax_img.set_axis_off()

  #Display histogram
  ax_hist.hist(image.ravel(), bins=bins)
  ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax_hist.set_xlabel('Pixel intensity')
  
  xmin, xmax = dtype_range[image.dtype.type]
  ax_hist.set_xlim(xmin, xmax)

  #Display cumulative distribution
  img_cdf, bins = exposure.cumulative_distribution(image, bins)
  ax_cdf.plot(bins, img_cdf, 'r')

  return ax_img, ax_hist, ax_cdf

img = img_as_ubyte(data.moon())
# 全局均衡化
img_rescale = exposure.equalize_hist(img)
# 局部均衡化
selem = disk(30)
img_eq = rank.equalize(img, selem=selem)

fig = plt.figure(figsize=(8,5))
axes = np.zeros((2,3), dtype=np.object)
axes[0,0] = plt.subplot(2,3,1, adjustable='box-forced')
axes[0,1] = plt.subplot(2,3,2, sharex=axes[0,0], sharey=axes[0,0], adjustable='box-forced')
axes[0,2] = plt.subplot(2,3,3, sharex=axes[0,0], sharey=axes[0,0], adjustable='box-forced')
axes[1,0] = plt.subplot(2,3,4)
axes[1,1] = plt.subplot(2,3,5)
axes[1,2] = plt.subplot(2,3,6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:,0])
ax_img.set_title('Low contrast image')
ax_hist.set_ylabel('Number of pixels')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:,1])
ax_img.set_title('Global equalise')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:,2])
ax_img.set_title('Local equalize')
ax_hist.set_ylabel('Fraction of total intensity')

fig.tight_layout()
plt.show()