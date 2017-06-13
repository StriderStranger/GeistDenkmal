#!/usr/bin/env python3
#-*- Skimage Gallery -- Adapting gray-scale filters to RGB images -*-
# 用装饰器实现多通道图像滤波
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

@adapt_rgb(each_channel)
def sobel_each(image):
  '''each_channel模式的彩色滤波器'''
  return filters.sobel(image)

@adapt_rgb(hsv_value)
def sobel_hsv(image):
  '''hsv_value模式的彩色滤波器'''
  return filters.sobel(image)

#--------------------------------------------------------------------------------#

from skimage import data
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

# 载入图片
image = data.astronaut()
fig = plt.figure(figsize=(14,7))
ax_each = fig.add_subplot(121, adjustable='box-forced')
ax_hsv = fig.add_subplot(122, sharex=ax_each, sharey=ax_each, adjustable='box-forced')

# sobel_each
ax_each.imshow(rescale_intensity(1 - sobel_each(image)))
ax_each.set_xticks([]), ax_each.set_yticks([])
ax_each.set_title('Sobel filter computed\n on individual RGB channels')

# sobel_hsv
ax_hsv.imshow(rescale_intensity(1 - sobel_hsv(image)))
ax_hsv.set_xticks([]), ax_hsv.set_yticks([])
ax_hsv.set_title('Sobel filter computed\n on converted image (HSV)')

#--------------------------------------------------------------------------------#
#自定义处理函数前两个参数必须是image_filter和image
def handler(image_filter, image, *args, **kwargs):
  '''创建自己的处理函数,以使用adapt_rgb装饰器'''
  #处理rgb图像在这里...
  image = image_filter(image, *args, **kwargs)
  #处理滤波后图像在这里...
  return image

#---------------------------------------------------------------------------------#

from skimage.color import rgb2gray

def as_gray(image_filter, image, *args, **kwargs):
  gray_image = rgb2gray(image)
  return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def sobel_gray(image):
  return filters.sobel(image)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111,sharex=ax_each, sharey=ax_each, adjustable='box-forced')
ax.imshow(rescale_intensity(1 - sobel_gray(image)), cmap=plt.cm.gray)
ax.set_xticks([]), ax.set_yticks([])
ax.set_title('Sobel filter computed\n on the converted grayscale image')



plt.show()
print('-0v0-')