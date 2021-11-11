from os import sep
import pyautogui
import cv2 #pip install opencv-contrib-python
import numpy as np
from time import sleep
from mss import mss
import timeit
from PIL import Image

decisionThreshold = .8
decisionThresholdCenter = .65

def resize(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

leftImg = cv2.imread('images/left_small.png');
rightImg = cv2.imread('images/right_small.png');
rightImgBig = cv2.imread('images/right_small.png');
leftImgBig = cv2.imread('images/left_small.png');
leftMiddleImg = cv2.imread('images/left_center.png');
rightMiddleImg = cv2.imread('images/right_center.png');
# leftMiddleImgGlow = cv2.imread('images/left_center_glow.png')
# rightMiddleImgGlow = cv2.imread('images/right_center_glow.png')

leftImg = cv2.cvtColor(leftImg,cv2.COLOR_RGB2GRAY)
rightImg = cv2.cvtColor(rightImg,cv2.COLOR_RGB2GRAY)
leftImgBig = cv2.cvtColor(leftImgBig,cv2.COLOR_RGB2GRAY)
rightImgBig = cv2.cvtColor(rightImgBig,cv2.COLOR_RGB2GRAY)
leftMiddleImg = cv2.cvtColor(leftMiddleImg,cv2.COLOR_RGB2GRAY)
rightMiddleImg = cv2.cvtColor(rightMiddleImg,cv2.COLOR_RGB2GRAY)
# leftMiddleImgGlow = cv2.cvtColor(leftMiddleImgGlow,cv2.COLOR_RGB2GRAY)
# rightMiddleImgGlow == cv2.cvtColor(rightMiddleImgGlow,cv2.COLOR_RGB2GRAY)

rightImgBig = resize(rightImgBig,145)
leftImgBig = resize(leftImgBig,145)
# leftMiddleImgGlow = resize(leftMiddleImgGlow,110)
# rightMiddleImgGlow  = resize(rightMiddleImg,110)
leftMiddleImg = resize(leftMiddleImg,100)
rightMiddleImg = resize(rightMiddleImg,100)
# resize(rightMiddleImg,50)

def checkNormalHit():
    sct = mss()
    im_bgr = np.array(sct.grab(monitor = {'top': 720, 'left': 400, 'width': 1100, 'height': 300}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2GRAY)  # 2

    # left
    result = cv2.matchTemplate(im_bgr,leftImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > decisionThreshold:
        pyautogui.press('left')
        return True
    # right
    result = cv2.matchTemplate(im_bgr,rightImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > decisionThreshold:
        pyautogui.press('right')
        return True
    result = cv2.matchTemplate(im_bgr,rightImgBig,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > decisionThreshold:
        pyautogui.press('right')
        return True
    result = cv2.matchTemplate(im_bgr,leftImgBig,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > decisionThreshold:
        pyautogui.press('left')
        # print("right struggle")
        return True

    return False
    
def checkMiddleHit():
    sct = mss() # 1315 400
    im_bgr = np.array(sct.grab(monitor = {'top': 170, 'left': 600, 'width': 1315 - 600 + 100, 'height': 400 - 170}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2GRAY)  # 2

    # left
    result = cv2.matchTemplate(im_bgr,leftMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result)

    # right
    result = cv2.matchTemplate(im_bgr,rightMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result)
    
    # print(max_val_right , max_val_left)
    # max_loc = max_loc_right
    # max_val = max_val_right
    # img = rightMiddleImg
    # x,y = max_loc
    # x1 ,y1 = x + img.shape[1] , y + img.shape[0]
    # cv2.rectangle(im_bgr , (x,y) , (x1,y1) , 255, 2)
    # cv2.putText(im_bgr, str(max_val), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    if max_val_right > decisionThresholdCenter and max_val_left > decisionThresholdCenter:
        pyautogui.press('left' if max_loc_left[1] > max_loc_right[1] else 'right')
        return True
    if max_val_left > decisionThresholdCenter:
        pyautogui.press('left')
        return True
    if max_val_right > decisionThresholdCenter:
        pyautogui.press('right')
        return True
    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    return False

while(True):
    if checkNormalHit(): continue
    if checkMiddleHit(): continue