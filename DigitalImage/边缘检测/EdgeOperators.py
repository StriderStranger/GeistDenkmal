#!/usr/bin/env python3
#-*- Skimage Gallery -- Edge Operators -*-
# 边缘操作(robert算子, sobel算子, scharr算子, prewitt算子)
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt

# image = camera()
# edge_roberts = roberts(image)
# edge_sobel = sobel(image)

# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8,4))
# ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
# ax[0].set_title('Roberts Edge Detection')
# ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
# ax[1].set_title('Sobel Edge Detection')
# for a in ax:
#   a.axis('off')
# plt.tight_layout()
# plt.show()

##===========================================================##
# 各个算子的区别

# 构造一张旋转不变的测试图像
x,y = np.ogrid[:100, :100]
img = np.exp(1j * np.hypot(x,y)**1.3 / 20.).real

edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

diff_scharr_prewitt = edge_scharr - edge_prewitt
diff_scharr_sobel = edge_scharr - edge_sobel
max_diff = np.max(np.maximum(diff_scharr_prewitt,diff_scharr_sobel))

fig, axes = plt.subplots(nrows=2,ncols=2, sharex=True, sharey=True, figsize=(8,8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_scharr, cmap=plt.cm.gray)
ax[1].set_title('Scharr Edge Detection')

ax[2].imshow(diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
ax[2].set_title('Scharr - Prewitt')

ax[3].imshow(diff_scharr_sobel, cmap=plt.cm.gray, vmax=max_diff)
ax[3].set_title('Scharr - Sobel')

for a in ax:
  a.axis('off')
plt.tight_layout()
plt.show()