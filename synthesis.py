import cv2
import numpy as np
from PIL import Image
import os

# 为方便调试,显示图片
def cvshowimg_debug(img, bWrite=False):
    cv2.imshow("image", img)
    if(bWrite):
        cv2.imwrite('./test.png', img)
    cv2.waitKey(0)

# 删除指定文件夹及其子文件夹下的所有文件 （不包含目录）
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

# 按照输入图片尺寸创建一个 nC 通道的全0图像
def np_zeros(img, nC=1):
    return np.zeros((img.shape[0], img.shape[1], nC), dtype=np.uint8)

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

def lblsave(filename, lbl, cmap):
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
        lbl_pil.putpalette(cmap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )

'''
功能: 按照指定角度 angle 旋转图片 img_in
方法流程: 
    1. 计算旋转矩阵
    2. 使用仿射变换实现图片旋转
参考: https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
'''
def rotate_img(img_in, angle, bPure=False):
    # 有透明通道,则保持透明通道不变
    if (img_in.shape[2] == 4):
        alpha_channel = img_in[:, :, 3]  # 获取透明通道

    # 1. 确定旋转矩阵
    (h, w) = img_in.shape[:2] # 图片分辨率
    center = (w // 2, h // 2) # 旋转中心
    # cv.getRotationMatrix2D(	center, angle, scale )  ->	retval
    # 其中 center为旋转中心, angle为旋转角度(非弧度/逆时针方向为正), scale为缩放因子
    # 参考: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. 通过仿射变换形状旋转
    # cv.warpAffine( src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]	) ->	dst
    # 其中 src	input image.
    #     dst	output image that has the size dsize and the same type as src .
    #     M	2×3 transformation matrix.
    #     dsize	size of the output image.
    #     flags	combination of interpolation methods (see InterpolationFlags)
    #           and the optional flag WARP_INVERSE_MAP that means that M is the inverse transformation ( dst→src ).
    #     borderMode	pixel extrapolation method (see BorderTypes); when borderMode=BORDER_TRANSPARENT,
    #                   it means that the pixels in the destination image corresponding to the "outliers"
    #                   in the source image are not modified by the function.
    #     borderValue	value used in case of a constant border; by default, it is 0.
    # 参考: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
    if(bPure):  # 修正插值, 返回单一像素值图片
        img_out = cv2.warpAffine(img_in, M, (w, h))
        pixel = img_in[center]
        ## To do!!!

    else:
        img_out = cv2.warpAffine(img_in, M, (w, h))

    # 有透明通道,则保持透明通道不变
    if (img_in.shape[2] == 4):
        img_out[:,:,3] = alpha_channel

    # cvshowimg_debug(img_out)
    return img_out

'''
功能: 将 img_fg 以中心对齐的方式叠加覆盖到 img_bg 上, blend表示混合系数，默认为0即无混合
方法流程: 
    1. 先做错误判断
    2. 计算边距
    3. 按照计算的ROI将img_fg赋值到img_bg 上
返回： 合成的图片 img_out
'''
def imgpaste_center(img_bg, img_fg, blend=0):
    assert (img_bg.shape[2] == img_fg.shape[2] or img_bg.shape[2]== 4 )
    rows_bg, cols_bg  = img_bg.shape[0], img_bg.shape[1]
    rows_fg, cols_fg = img_fg.shape[0], img_fg.shape[1]
    assert ( cols_fg <= cols_bg and rows_fg <= rows_bg)

    img_out = img_bg.copy()
    # 计算边距
    space_w = (cols_bg - cols_fg) // 2
    space_h = (rows_bg - rows_fg) // 2
    if(blend>0): # 混合模式且不存在透明通道
        if (blend>1):
            blend = 0.99
        roi = img_bg[space_w:space_w + cols_fg, space_h:space_h + rows_fg]  # 取出背景ROI区域
        tmp = cv2.addWeighted(roi, blend, img_fg, 1 - blend, 0)  # 加权混合
        img_out[space_w:space_w + cols_fg, space_h:space_h + rows_fg] = tmp
    else:
        if(img_bg.shape[2] == img_fg.shape[2]):   # 背景不含义透明通道
            img_out[space_h:space_h + rows_fg, space_w:space_w + cols_fg] = img_fg
        else:   # 含有透明通道 等通道数不等的情况
            img_out[space_h:space_h + rows_fg, space_w:space_w + cols_fg, 0:3] = img_fg[:, :, 0:3]

    # cvshowimg_debug(img_out)
    return img_out

'''
功能: 将 img_tube 按照行列号叠加覆盖到 img_bg 上， img_tube的透明通道部分不能进行覆盖
方法流程: 
    1. 先做错误判断
    2. 
返回： 合成的图片 img_out
'''
def imgpaste_rack(img_bg, img_tube, posRow, posCol):
    assert (img_tube.shape[2] == 4 and img_bg.shape[2]== 3 )    # tube 图片带透明通道
    rows_bg, cols_bg  = img_bg.shape[0], img_bg.shape[1]
    rows_tube, cols_tube = img_tube.shape[0], img_tube.shape[1]
    assert ( cols_tube <= cols_bg and rows_tube <= rows_bg)

    x_start, y_start = 136, 154 # 第一个试管位的中心点坐标
    x_end, y_end = 1222, 822
    deltaY = posRow*(y_end-y_start)/7.0 # 行列中心点差 99 个像素
    deltaX = posCol*(x_end-x_start)/11.0

    # 计算边距
    img_out = img_bg.copy()
    space_left = round(x_start+deltaX) - cols_tube // 2
    space_top = round(y_start+deltaY) - rows_tube // 2

    alpha_channel = img_tube[:, :, 3] / 255.0 # 透明通道信息， 0为透明， 1为不透明
    alpha_channel_3 = np.zeros((rows_tube, cols_tube, 3), np.uint8)
    alpha_channel_3[:,:,0] = alpha_channel
    alpha_channel_3[:, :, 1] = alpha_channel
    alpha_channel_3[:, :, 2] = alpha_channel

    roi = img_bg[space_top:space_top+rows_tube, space_left:space_left+cols_tube]  # 取出背景ROI区域
    test = (1-alpha_channel_3)*roi+alpha_channel_3*img_tube[:, :, 0:3]
    img_out[space_top:space_top+rows_tube, space_left:space_left+cols_tube] = test

    # cvshowimg_debug(img_out)
    return img_out

def imgpaste_rack_mask(img_bg_zero, img_tube_mask, posRow, posCol):
    rows_bg, cols_bg  = img_bg_zero.shape[0], img_bg_zero.shape[1]
    rows_tube, cols_tube = img_tube_mask.shape[0], img_tube_mask.shape[1]
    assert ( cols_tube <= cols_bg and rows_tube <= rows_bg)

    x_start, y_start = 136, 154 # 第一个试管位的中心点坐标
    x_end, y_end = 1224, 825
    deltaY = posRow*(y_end-y_start)/7.0 # 行列中心点差 99 个像素
    deltaX = posCol*(x_end-x_start)/11.0

    # 计算边距
    img_out_mask = img_bg_zero.copy()
    space_left = round(x_start+deltaX) - cols_tube // 2
    space_top = round(y_start+deltaY) - rows_tube // 2

    img_out_mask[space_top:space_top+rows_tube, space_left:space_left+cols_tube, 0] = img_tube_mask

    #cvshowimg_debug(img_out_mask)
    return img_out_mask