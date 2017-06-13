#!/usr/bin/env python3
#-*- Skimage Gallery -- Circular and Elliptical Hough transform -*-
# 霍夫变换检测圆和椭圆
# 圆检测: 给出可能的半径范围,来寻找圆心,响应大的即为圆心的位置
# 椭圆检测: 取两点作为主轴线端点,在通过其他点响应
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

image = img_as_ubyte(data.coins()[160:230, 70:270])
edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

# hough_radii是可能的半径列表
hough_radii = np.arange(20, 35, 2)
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

fig, ax = plt.subplots(1,1,figsize=(10,4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy,cx,radii):
  circley,circlex = circle_perimeter(center_y,center_x, radius)   # 将圆心半径转成圆周坐标
  image[circley,circlex] = (220,20,20)
ax.imshow(image, cmap=plt.cm.gray)
# plt.show()

from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
image_rgb = data.coffee()[0:220, 160:420]
image_gray = color.rgb2gray(image_rgb)
edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

# accuracy: 主轴的bin size
# threshold: 去掉响应低的情况
result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
result.sort(order='accumulator')

best = list(result[-1])
yc,xc,a,b = [int(round(x)) for x in best[1:5]]    # 选择排序后的前五个结果
orientation = best[5]

cy,cx = ellipse_perimeter(yc,xc,a,b,orientation)
image_rgb[cy,cx] = (0,0,255)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy,cx] = (250,0,0)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax1.set_title('Original picture')
ax1.imshow(image_rgb)
ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)
plt.show()
print('result:',result[0])