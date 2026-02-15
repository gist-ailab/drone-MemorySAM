import cv2
import numpy as np

pth = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/zed/lj2_0_031875.png'
img = cv2.imread(pth)
print(img.shape)
print(img.dtype)
print(img.max())
print(img.min())
print(img.mean())
print(img.std())
print(img.sum())
print(img.size)
print(img.ndim)
print(img.nbytes)