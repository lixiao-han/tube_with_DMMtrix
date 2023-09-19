import cv2

img = cv2.imread('./data/bg/bg_tube.png',cv2.IMREAD_UNCHANGED)

print(img.shape[0], img.shape[1])