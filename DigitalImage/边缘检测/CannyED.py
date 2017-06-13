#!/usr/bin/env python3
#-*- Skimage Gallery -- Canny Edge Detector -*-
# Canny边界检测算子
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature

# 生成测试图像
im = np.zeros((128,128))
im[32:-32, 32:-32] = 1
im = ndi.rotate(im, 15, mode='constant')
im += 0.2 * np.random.random(im.shape)

# 加和不加高斯去噪的canny
edge1 = feature.canny(im)
edge2 = feature.canny(im, sigma=3)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8,3), sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edge1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edge2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()
plt.show()