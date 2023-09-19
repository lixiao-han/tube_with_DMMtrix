import os, random, shutil
import cv2
import argparse
import numpy as np
from math import *
import os.path as osp
import PIL
from PIL import Image


def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def selectfile(input_imgs_dir, output_imgs_dir, num):
    images = os.listdir(input_imgs_dir)
    sample = random.sample(images, num)
    for name in sample:
        shutil.copyfile(input_imgs_dir + name, output_imgs_dir + name)
    return

def lblsave(filename, lbl, cmap):
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl, mode='P')
        lbl_pil.putpalette(cmap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % filename
        )

def transfer(cmap):
    # 读入图片并将opencv的BGR转换为RGB格式
    img = cv2.imread("E:/plant20211210/20211210first/data/mask/0.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape[2])
    # 将24位深图片中的[r,g,b]对应到colormap反求出label
    lbls = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    len_cmap = len(cmap)
    for i1 in range(len_cmap):
        print(i1, cmap[i1])
    print('接下来是')
    indexes = np.nonzero(img)
    print(indexes[0])
    print(indexes[1])
    for i, j in zip(indexes[0], indexes[1]):
        for k in range(len_cmap):
            if img[i, j, 0] == cmap[k, 0] and img[i, j, 1] == cmap[k, 1] and img[i, j, 2] == cmap[k, 2]:
                lbls[i, j] = k
                break

    cv2.imwrite("E:/plant20211210/20211210first/data/mask/1.png", lbls)
    #使用了调色盘的图像将会被单通道存储，每个像素位置的值是调色盘“表”的索引，这在存储图像的时候空间要求从RGB的3个字节变成了一个字节
    lblsave(os.path.join("E:/plant20211210/20211210first/data/mask/2.png"), lbls, cmap)

def mask_white(output_imgs_dir, num):
    filestr = os.listdir(output_imgs_dir)

    for i in range(num):
        filename = filestr[i]
        img = cv2.imread(output_imgs_dir + filename, cv2.IMREAD_UNCHANGED)
        w = img.shape[1]
        h = img.shape[0]
        mask = np.zeros((h, w, 3), np.uint8)
        mask[0:h, 0:w, 0:3] = 255
        cv2.imwrite(mask_white_dir + filename, mask)


# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_color_bg(combine_imgs_dir, mask_combine_dir, num):
    # grab the dimensions of the image and then determine the
    # center
    filestr = os.listdir(combine_imgs_dir)
    filestr1 = os.listdir(mask_combine_dir)

    for i in range(num):
        filename = filestr[i]
        filename1 = filestr1[i]
        image = cv2.imread(combine_imgs_dir + filename, cv2.IMREAD_UNCHANGED)
        image1 = cv2.imread(mask_combine_dir + filename1, cv2.IMREAD_UNCHANGED)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
        M = cv2.getRotationMatrix2D((cX, cY), -10, 1.0)

        #cos = np.abs(M[0, 0])
        #sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
       # nW = int((h * sin) + (w * cos))
        #nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        #M[0, 2] += (nW / 2) - cX
        #M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        # borderValue 缺失背景填充色彩，此处为白色，可自定义
        imageRotation = cv2.warpAffine(image, M, (w, h))
        imageRotation1 = cv2.warpAffine(image1, M, (w, h))

        # borderValue 缺省，默认是黑色（0, 0 , 0）
        cv2.imwrite(rotate_imgs_dir + filename, imageRotation)
        cv2.imwrite(mask_rotate_dir + filename1, imageRotation1)


def mask_color(mask_white_dir, num):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    global cmap
    filestr1 = os.listdir(mask_white_dir)
    cmap = np.zeros((num+1, 3), dtype=np.uint8)
    cmap[num, 0] = 255
    cmap[num, 1] = 255
    cmap[num, 2] = 255
    value = None
    for i in range(num):
        filename1 = filestr1[i]
        image1 = cv2.imread(mask_white_dir + filename1, cv2.IMREAD_UNCHANGED)
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        print("染色-",i,":[",r,g,b,"]")
        # 染色
        indexes = np.nonzero(image1)
        for ii, jj in zip(indexes[0], indexes[1]):
            image1[ii, jj, 0] = b;
            image1[ii, jj, 1] = g;
            image1[ii, jj, 2] = r;
        # 对应索引为 i 的调色板值

        cv2.imwrite(mask_color_dir + filename1, image1)
        thecolor = cv2.imread(mask_color_dir + filename1, cv2.IMREAD_UNCHANGED)
        cmap[i, 0] = thecolor[10, 10, 2]
        cmap[i, 1] = thecolor[10, 10, 1]
        cmap[i, 2] = thecolor[10, 10, 0]
        print("cmap-", i, ":[", r, g, b, "]//")



def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new

def imgpaste(mother_img1, rotate_imgs_dir, num):
    M_Img = cv2.imread(mother_img1,cv2.IMREAD_UNCHANGED)
    w1 = M_Img.shape[0]
    h1 = M_Img.shape[1]
    for hhh in range(h1):
        for www in range(w1):
            if M_Img[hhh, www, 0] == 255:
                M_Img[hhh, www, 3]=0
    M_Img1 = np.zeros((h1, w1, 3), np.uint8)
    filestr = os.listdir(rotate_imgs_dir)
    filestr1 = os.listdir(mask_color_dir)
    for count in range(num):
        filename = filestr[count]
        filename1 = filestr1[count]
        S_Img = cv2.imread(rotate_imgs_dir + filename, cv2.IMREAD_UNCHANGED)
        S_Img = cv2.resize(S_Img, (30, 30))
        S_Img1 = cv2.imread(mask_color_dir + filename1, cv2.IMREAD_UNCHANGED)
        S_Img1 = cv2.resize(S_Img1, (30, 30))
        img = S_Img.shape
        w = img[0]
        h = img[1]
        x0 = int(12)  # 第一个dm码左上角横坐标
        y0 = int(12)  # 第一个dm码左上角纵坐标
        x1 = x0 + w  # 第一个dm码右下角横坐标
        y1 = y0 + h  # 第一个dm码右下角纵坐标
        for c in range(0, 3):
            M_Img[y0:y1, x0:x1, c] =  S_Img[:, :, c]
        for c in range(0, 3):
            M_Img1[y0:y1, x0:x1, c] =  S_Img1[:, :, c]
        cv2.imwrite(combine_imgs_dir + str(count) +".png", M_Img)
        cv2.imwrite(mask_combine_dir + str(count) +".png", M_Img1)
        M_Img = cv2.imread(mother_img1, cv2.IMREAD_UNCHANGED)
        M_Img1 = np.zeros((h1, w1, 3), np.uint8)
        for hhh in range(h1):
            for www in range(w1):
                if M_Img[hhh, www, 0] == 255:
                    M_Img[hhh, www, 3] = 0


def imagepaste(mother_img, rotate_imgs_dir, num):
    M_Img = cv2.imread(mother_img)
    ones = np.ones((M_Img.shape[0], M_Img.shape[1])) * 255
    M_Img = np.dstack([M_Img, ones])
    h1 = M_Img.shape[0]
    w1 = M_Img.shape[1]
    M_Img1 = np.zeros((h1, w1, 3), np.uint8)
    filestr = os.listdir(rotate_imgs_dir)
    filestr1 = os.listdir(mask_rotate_dir)
    posList = []
    for count in range(num):
        filename = filestr[count]
        filename1 = filestr1[count]
        S_Img = cv2.imread(rotate_imgs_dir + filename, cv2.IMREAD_UNCHANGED)
        S_Img1 = cv2.imread(mask_rotate_dir + filename1, cv2.IMREAD_UNCHANGED)
        img = S_Img.shape
        w = img[0]
        h = img[1]
        l = np.random.randint(1, 12)  # 列数
        m = np.random.randint(1, 8)  # 行数
        while [m, l] in posList:
            m, l = random.randint(1, 8), random.randint(1, 12)
        posList.append([m, l])
        alpha_image_3 = S_Img[:, :, 3] / 255.0
        alpha_image = 1 - alpha_image_3
        if l <= 6 and m <= 4:
            x0 = int(106)
            y0 = int(123)
            x1 = x0 + w
            y1 = y0 + h
            x2 = x0 + (l - 1) * 99
            y2 = y0 + (m - 1) * 97
            x3 = x1 + (l - 1) * 99
            y3 = y1 + (m - 1) * 97
            for c in range(0, 3):
                M_Img[y2:y3, x2:x3, c] = ((alpha_image * M_Img[y2:y3, x2:x3, c]) + (alpha_image_3 * S_Img[:, :, c]))
            for c in range(0, 3):
                M_Img1[y2:y3, x2:x3, c] =S_Img1[:, :, c]
        if l > 6 and m <= 4:
            x0 = int(699)
            y0 = int(127)
            x1 = x0 + w
            y1 = y0 + h
            x2 = x0 + (l - 7) * 98
            y2 = y0 + (m - 1) * 95
            x3 = x1 + (l - 7) * 98
            y3 = y1 + (m - 1) * 95
            for c in range(0, 3):
                M_Img[y2:y3, x2:x3, c] = ((alpha_image * M_Img[y2:y3, x2:x3, c]) + (alpha_image_3 * S_Img[:, :, c]))
            for c in range(0, 3):
                M_Img1[y2:y3, x2:x3, c] = S_Img1[:, :, c]
        if l <= 6 and m > 4:
            x0 = int(110)
            y0 = int(508)
            x1 = x0 + w
            y1 = y0 + h
            x2 = x0 + (l - 1) * 98
            y2 = y0 + (m - 5) * 95
            x3 = x1 + (l - 1) * 98
            y3 = y1 + (m - 5) * 95
            for c in range(0, 3):
                M_Img[y2:y3, x2:x3, c] = ((alpha_image * M_Img[y2:y3, x2:x3, c]) + (alpha_image_3 * S_Img[:, :, c]))
            for c in range(0, 3):
                M_Img1[y2:y3, x2:x3, c] = S_Img1[:, :, c]
        if l > 6 and m > 4:
            x0 = int(699)
            y0 = int(508)
            x1 = x0 + w
            y1 = y0 + h
            x2 = x0 + (l - 7) * 99
            y2 = y0 + (m - 5) * 95
            x3 = x1 + (l - 7) * 99
            y3 = y1 + (m - 5) * 95
            for c in range(0, 3):
                M_Img[y2:y3, x2:x3, c] = ((alpha_image * M_Img[y2:y3, x2:x3, c]) + (alpha_image_3 * S_Img[:, :, c]))
            for c in range(0, 3):
                M_Img1[y2:y3, x2:x3, c] = S_Img1[:, :, c]
        cv2.imwrite(result_imgs_dir, M_Img)
        cv2.imwrite(mask_result_dir, M_Img1)


if __name__ == "__main__":
    input_imgs_dir = "E:/plant20211210/20211210first/data/dmcode_syn/"
    output_imgs_dir = "E:/plant20211210/20211210first/data/sample/"
    rotate_imgs_dir = "E:/plant20211210/20211210first/data/rotate/"
    mother_img = "E:/plant20211210/20211210first/data/bg/bg_rack.png"
    mother_img1 = "E:/plant20211210/20211210first/data/bg/bg_tube.png"
    combine_imgs_dir = "E:/plant20211210/20211210first/data/combine/"
    result_imgs_dir = "E:/plant20211210/20211210first/data/0.png"
    mask_white_dir = "E:/plant20211210/20211210first/data/mask/mask_white/"
    mask_rotate_dir = "E:/plant20211210/20211210first/data/mask/mask_rotate/"
    mask_combine_dir = "E:/plant20211210/20211210first/data/mask/mask_combine/"
    mask_color_dir = "E:/plant20211210/20211210first/data/mask/mask_color/"
    mask_result_dir = "E:/plant20211210/20211210first/data/mask/0.png"
    if not os.path.exists(output_imgs_dir):
        os.mkdir(output_imgs_dir)
    num = random.randint(1, 96)
    selectfile(input_imgs_dir, output_imgs_dir, num)
    mask_white(output_imgs_dir, num)
    mask_color(mask_white_dir, num)
    imgpaste(mother_img1, output_imgs_dir, num)
    rotate_bound_color_bg(combine_imgs_dir, mask_combine_dir, num)
    imagepaste(mother_img, rotate_imgs_dir, num)
    transfer(cmap)
