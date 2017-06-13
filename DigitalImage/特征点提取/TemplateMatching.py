#!/usr/bin/env python3
#-*- Skimage Gallery -- Template Matching -*-
# 使用"Fast Normalized Cross-Correlation"算法进行模型匹配
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import match_template

image = data.coins()
coin = image[170:220, 75:130]

# result是image关于coin的响应结果
# 找到result中取值最大的点的坐标(x,y)
result = match_template(image, coin)
ij = np.unravel_index(np.argmax(result), result.shape)
x,y = ij[::-1]

fig = plt.figure(figsize=(8,3))
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2, adjustable='box-forced')
ax3 = plt.subplot(1,3,3, sharex=ax2, sharey=ax2, adjustable='box-forced')
ax1.imshow(coin, cmap=plt.cm.gray)
ax1.set_title('template')
ax1.set_axis_off()
ax2.imshow(image, cmap=plt.cm.gray)
ax2.set_title('image')
ax2.set_axis_off()
hcoin,wcoin = coin.shape
rect = plt.Rectangle((x,y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)     # 在原图上添加标记
ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('match_template\nresult')
ax3.autoscale(False)
ax3.plot(x,y,'o',markeredgecolor='r', markerfacecolor='none', markersize=10)
plt.show()