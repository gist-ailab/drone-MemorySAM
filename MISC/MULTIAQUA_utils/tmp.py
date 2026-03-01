import cv2
import numpy as np
import matplotlib.pyplot as plt
# pth = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed/lj2_0_031875.png'
pth = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera/lj2_0_031875.png'
img = cv2.imread(pth)


plt.imshow(img)
plt.show()