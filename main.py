import random
import cv2
import sys

## 合成一张训练图片
# bFull 为真时,合成含96个DM码的图片, 采用 ./dmcode_syn 中的DM码
# 为假时,合成随机个树的DM码,
def run(bFull=True):
    # print('Hello Test')
    # 读取背景图像
    img_bg_rack = cv2.imread("./data/bg/bg_rack.png")
    img_bg_tube = cv2.imread("./data/bg/bg_tube.png")

    # 判断合成模式
    if bFull != True:
        numDMCodes = random.randint(1,96)   # 随机DM码数量：numDMCodes
    else:
        numDMCodes = 96
    posList = [] # 记录所有DM码行列号的列表
    for index in range(numDMCodes):
        if bFull != True:
            posRow, posCol = random.randint(0,7), random.randint(0,11)    # 随机第index个DM码的位置：posRow、posCol
            while [posRow, posCol] in posList:
                posRow, posCol = random.randint(0, 7), random.randint(0, 11)
        else:
            posRow, posCol = (int)(index / 12), index % 12
        posList.append([posRow, posCol])
        #--------------------合成DM码部分--------------------#
        # 随机读取（生成）一个DM码 img_dm

        # 随机旋转该DM码 img_dm_rot

        # 将img_dm_rot添加到img_bg_tube上合成dm_tube

        # 将dm_tube按照（posRow, posCol）位置添加到img_bg_rack上得到 img_syn


        #--------------------合成DM码对应的mask部分--------------------#
        # 绘制与img_dm区域同等大小的矩形，并以 index+1 进行填充，得到 img_dm_mask

        # 对 img_dm_mask 进行同 img_dm_rot 一致的旋转得到 img_dm_mask_rot

        # 将 img_dm_mask_rot 添加到 img_bg_tube 上合成 dm_tube_mask

        # 将 dm_tube_mask 按照（posRow, posCol）位置添加到img_bg_rack上得到 img_syn_mask

        # 生成调色板 cmap

    return img_syn, img_syn_mask


if __name__ == '__main__':
    #count = sys.argv[1]
    count = 10
    bFull = True
    #bFull = False
    for var in range(count): #生成count张合成图片
        img_syn, img_syn_mask = run(bFull)
        # 生成对应的 info.yaml并保存

        # 保存 img_syn为png图片，以 dm{count}.png 命名

        # 依据 cmap的颜色映射将 img_syn_mask 保存为8位带颜色的png图片