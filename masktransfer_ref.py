import cv2
import numpy as np
import os.path as osp
import os
from PIL import Image

def label_colormap(n_label=256, value=None):
    """Label colormap.

    Parameters
    ----------
    n_labels: int
        Number of labels (default: 256).
    value: float or int
        Value scale or value of label color in HSV space.

    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.

    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        hsv = cv2.rgb2hsv(cmap.reshape(1, -1, 3))
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        cmap = cv2.hsv2rgb(hsv).reshape(-1, 3)
    return cmap


def lblsave(filename, lbl):
    if osp.splitext(filename)[1] != '.png':
        filename += '.png'
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
        colormap = label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )

## 获取颜色映射表
colormap = label_colormap()

from random import *
pos = randint(0,255)
print(colormap[pos,0],colormap[pos,1],colormap[pos,2])


# 读入图片并将opencv的BGR转换为RGB格式
img = cv2.imread("1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将24位深图片中的[r,g,b]对应到colormap反求出label
lbls = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.int32)
len_colormap = len(colormap)
indexes = np.nonzero(img)

for i, j in zip(indexes[0], indexes[1]):
    for k in range(len_colormap):
        if all(img[i, j, :3] == colormap[k]):
            lbls[i, j] = k
            break

#####
##### 此处添加对lal图的变换处理过程，注意数据类型为int32
#####

# 将label再转换成8位label.png
lblsave(os.path.join(os.getcwd(), 'label1-1.png'), lbls)