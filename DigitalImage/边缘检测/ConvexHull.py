#!/usr/bin/env python3
#-*- Skimage Gallery -- Convex Hull -*-
# 二值图像的最小凸包
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert

# image = data.horse()
image = invert(data.horse())
chull = convex_hull_image(image)

fig, axes = plt.subplots(1,2, figsize=(8,4))
ax = axes.ravel()

ax[0].set_title('Original picture')
ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_axis_off()

ax[1].set_title('Transformed picture')
ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_axis_off()

plt.tight_layout()
plt.show()