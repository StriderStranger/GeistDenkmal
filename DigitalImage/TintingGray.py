#!/usr/bin/env python3
#-*- Skimage Gallery -- Tinting gray-scale images -*-
# 通道操作
import matplotlib.pyplot as plt
from skimage import data
from skimage import color
from skimage import img_as_float

grayscale_image = img_as_float(data.camera()[::2,::2])
image = color.gray2rgb(grayscale_image)

red_multiplier = [1,0,0]
yellow_multiplier = [1,1,0]

fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8,4), sharex=True, sharey=True)
# 保留R通道
ax1.imshow(red_multiplier * image)
# 保留RG通道
ax2.imshow(yellow_multiplier * image)
ax1.set_adjustable('box-forced')
ax2.set_adjustable('box-forced')

# Plot H通道的梯度图
import numpy as np
hue_gradient = np.linspace(0,1)
hsv = np.ones(shape=(1, len(hue_gradient), 3), dtype=float)
hsv[:,:,0] = hue_gradient
all_hues = color.hsv2rgb(hsv)
fig,ax = plt.subplots(figsize=(5,2))
ax.imshow(all_hues, extent=(0, 1, 0, 0.2))
ax.set_axis_off()

def colorize(image, hue, saturation=1):
  '''rgb -> hsv(修改) -> rgb'''
  hsv = color.rgb2hsv(image)
  hsv[:,:,1] = saturation
  hsv[:,:,0] = hue
  return color.hsv2rgb(hsv)

# S通道固定1,H通道固定6个值
hue_rotations = np.linspace(0,1,6)
fig,axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
for ax, hue in zip(axes.flat, hue_rotations):
  tinted_image = colorize(image, hue, saturation=0.3)
  ax.imshow(tinted_image, vmin=0, vmax=1)
  ax.set_axis_off()
  ax.set_adjustable('box-forced')
fig.tight_layout()


# 局部区域通道变化
from skimage.filters import rank
# 1.用slice构建区域
top_left = (slice(100),) * 2                        # (:100, :100)
bottom_right = (slice(-100,None),) * 2              # (:-100, :-100)
sliced_image = image.copy()
sliced_image[top_left] = colorize(image[top_left], 0.82, saturation=0.5)
sliced_image[bottom_right] = colorize(image[bottom_right], 0.5, saturation=0.5)
# 2.构造mask
noisy = rank.entropy(grayscale_image, np.ones((9,9)))
textured_regions = noisy > 4
masked_image = image.copy()
masked_image[textured_regions, :] *= red_multiplier

fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1, figsize=(8,4), sharex=True, sharey=True)
ax1.imshow(sliced_image)
ax2.imshow(masked_image)
ax1.set_adjustable('box-forced')
ax2.set_adjustable('box-forced')

plt.show()