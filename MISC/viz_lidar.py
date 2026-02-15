import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
from torchvision import io

pth = '/ailab_mat2//dataset/DELIVER/lidar/fog/test/MAP_7_point4/006250_lidar_front.png'

img = cv2.imread(pth)
minmax = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
print(img.shape)




mask_pth = '/ailab_mat2//dataset/DELIVER/semantic/fog/test/MAP_7_point4/006250_semantic_front.png'

# DELIVER 데이터셋의 PALETTE 정의
PALETTE = torch.tensor([[70, 70, 70],
        [100, 40, 40],
        [55, 90, 80],
        [220, 20, 60],
        [153, 153, 153],
        [157, 234, 50],
        [128, 64, 128],
        [244, 35, 232],
        [107, 142, 35],
        [0, 0, 142],
        [102, 102, 156],
        [220, 220, 0],
        [70, 130, 180],
        [81, 0, 81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180],
        [250, 170, 30],
        [110, 190, 160],
        [170, 120, 50],
        [45, 60, 150],
        [145, 170, 100],
        [  0,  0, 230], 
        [  0, 60, 100],
        [  0,  0, 70],
        ], dtype=torch.uint8)

# 마스크 읽기
mask = io.read_image(mask_pth)[0, ...]  # (H, W) 형태
print(f"Mask shape: {mask.shape}")
print(f"Mask unique values: {torch.unique(mask)}")

# 마스크 전처리 (deliver.py와 동일하게)
mask_processed = mask.clone()
mask_processed[mask_processed == 255] = 0
mask_processed = mask_processed - 1
mask_processed[mask_processed < 0] = 0  # 음수 방지

# PALETTE를 사용하여 색칠된 마스크 생성
H, W = mask_processed.shape
colored_mask = torch.zeros((H, W, 3), dtype=torch.uint8)

for class_id in range(len(PALETTE)):
    mask_indices = (mask_processed == class_id)
    if mask_indices.any():
        colored_mask[mask_indices] = PALETTE[class_id]

# BGR로 변환 (OpenCV 저장용)
colored_mask_bgr = colored_mask.numpy()[:, :, ::-1]  # RGB -> BGR



# 디렉토리가 없으면 생성
output_path = 'visualized_mask.png'
# 이미지 저장
cv2.imwrite(output_path, colored_mask_bgr)
print(f"Colored mask saved to: {output_path}")

# matplotlib으로도 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mask.numpy(), cmap='gray')
plt.title('Original Mask')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(colored_mask.numpy())
plt.title('Colored Mask')
plt.axis('off')

plt.tight_layout()
viz_output_path = output_path.replace('.png', '_viz.png')
plt.savefig(viz_output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {viz_output_path}")
plt.close()