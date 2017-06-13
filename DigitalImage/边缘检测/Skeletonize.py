#!/usr/bin/env python3
#-*- Skimage Gallery -- Seletonize -*-
# 骨骼化: 一层层去除边界直到宽度为1
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

image = invert(data.horse())
skeleton = skeletonize(image)

fig, axes = plt.subplots(1,2, figsize=(8,4), sharex=True,sharey=True, subplot_kw={'adjustable':'box-forced'})
ax = axes.ravel()
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)
ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)
plt.tight_layout()
plt.show()