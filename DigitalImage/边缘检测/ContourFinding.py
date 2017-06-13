#!/usr/bin/env python3
#-*- Skimage Gallery -- Contour Finding -*-
# 用 marching squares算法寻找轮廓边界
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# 构造一张测试图
x,y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# 寻找边界,返回插值后的坐标
contours = measure.find_contours(r, 0.8)

fig, ax = plt.subplots()
ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

for n,contours in enumerate(contours):
  ax.plot(contours[:,1], contours[:,0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()