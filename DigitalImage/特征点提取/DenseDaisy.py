#!/usr/bin/env python3
#-*- Skimage Gallery -- Dense DAISY feature description -*-
# DAISY特征描述子
from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt
img = data.camera()

descs,descs_img = daisy(img, step=180, radius=58, rings=2, histograms=6,
                        orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(descs_img)
descs_num = descs.shape[0] * descs.shape[1]
ax.set_title('%i DAISY descriptors extracted:' % descs_num)
print(descs.shape)
plt.show()