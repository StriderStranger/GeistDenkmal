#!/usr/bin/env python3
#-*- Skimage Gallery -- Multi-Block Local Binary Pattern for texture classification -*-
# MBLBP特征提取算法 描述纹理特征(旋转不变性,灰度不变性)
from skimage.feature import multiblock_lbp
import numpy as np
from numpy.testing import assert_equal
from skimage.transform import integral_image
import matplotlib.pyplot as plt

test_img = np.zeros((9,9), dtype='uint8')
test_img[3:6, 3:6] = 1
test_img[:3, :3] = 50
test_img[6:, 6:] = 50
correct_answer = 0b10001000
int_img = integral_image(test_img)
lbp_code = multiblock_lbp(int_img, 0,0,3,3)
assert_equal(correct_answer, lbp_code)

from skimage import data
from skimage.feature import draw_multiblock_lbp
test_img = data.coins()

int_img = integral_image(test_img)
lbp_code = multiblock_lbp(int_img, 0,0,90,90)
img = draw_multiblock_lbp(test_img, 0,0,90,90, lbp_code=lbp_code, alpha=0.5)

plt.imshow(img, interpolation='nearest')
plt.show()
print(lbp_code)