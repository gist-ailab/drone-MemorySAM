import numpy as np
import cv2
import matplotlib.pyplot as plt
root = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera'


import os
import glob

# Directory with thermal images
thermal_dir = '/ailab_mat2/personal/jemo_maeng/dset/Drone/MULTIAQUA_night/MULTIAQUA_night/data/thermal_camera'
image_files = sorted(glob.glob(os.path.join(thermal_dir, '*.png')))

if not image_files:
    print('No PNG images found in the specified directory.')
    exit(1)

idx = 0
num_images = len(image_files)
win_name = 'CLAHE Thermal Viewer'

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def show_img(idx):
    img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        clahe_img = clahe.apply(img)
        display = cv2.hconcat([
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
        ])
        cv2.putText(display, f'{os.path.basename(image_files[idx])} ({idx+1}/{num_images})',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow(win_name, display)
    else:
        print(f"Failed to load image: {image_files[idx]}")

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
show_img(idx)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord('d') or key == 83:  # right arrow or 'd'
        idx = (idx + 1) % num_images
        show_img(idx)
    elif key == ord('a') or key == 81:  # left arrow or 'a'
        idx = (idx - 1 + num_images) % num_images
        show_img(idx)

cv2.destroyAllWindows()