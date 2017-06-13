#!/usr/bin/env python3
#-*- Skimage Gallery -- Histogram of Oriented Gradients -*-
# HOG特征描述子
# 1.全局图像正则化
# 2.计算x，y方向的梯度
# 3.计算梯度直方图（基于方向）
# 4.局部blocks正则化
# 5.展开成特征向量
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
image = color.rgb2gray(data.astronaut())

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

                    
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax2.set_adjustable('box-forced')
print(fd.shape)
plt.show()