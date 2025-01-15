import pdb

import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from PIL import Image


img_path = "/home/buaa/Desktop/frame_30/demo_2.jpg"
attn9_path = "/home/buaa/Desktop/frame_30/demo_2_layer9.npy"
attn12_path = "/home/buaa/Desktop/frame_30/demo_2_layer12.npy"

attn9 = np.load(attn9_path)
attn12 = np.load(attn12_path)


image = Image.open(img_path)
image = np.array(image).transpose(2, 0, 1)  # 转换为 [3, 256, 256]

# 确保图像数据类型为 float32
image = image.astype(np.float32)

# 归一化图像数据到 [0, 1] 范围内
image /= 255.0

# 将通道维度从第一个位置移动到最后一个位置以适应 imshow
image = np.moveaxis(image, 0, -1)

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(4,12), dpi=300)
ax1.imshow(image)
ax1.axis('off')

ax2.imshow(attn9, cmap='coolwarm')
ax2.axis('off')

ax3.imshow(attn12, cmap='coolwarm')
ax3.axis('off')

plt.axis('off')

# 保存热力图为文件
output_filename = '/home/buaa/Desktop/heatmap_demo_2_new.png'
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)

# 显示热力图（可选）
# plt.show()

# plt.savefig(output_filename)
plt.close()