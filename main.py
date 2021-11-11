import pyautogui
import cv2 #pip install opencv-contrib-python
import numpy as np
from time import sleep
from mss import mss
import timeit
from PIL import Image

decisionThreshold = .8
decisionThresholdCenter = .8

def resize(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

leftImg = cv2.imread('images/left_small.png');
rightImg = cv2.imread('images/right_small.png');
rightImgBig = cv2.imread('images/right_small.png');
leftImgBig = cv2.imread('images/left_small.png');
leftMiddleImg = cv2.imread('images/left_center_tip2.png');
rightMiddleImg = cv2.imread('images/right_center_tip2.png');

leftImg = cv2.cvtColor(leftImg,cv2.COLOR_RGB2GRAY)
rightImg = cv2.cvtColor(rightImg,cv2.COLOR_RGB2GRAY)
leftImgBig = cv2.cvtColor(leftImgBig,cv2.COLOR_RGB2GRAY)
rightImgBig = cv2.cvtColor(rightImgBig,cv2.COLOR_RGB2GRAY)
leftMiddleImg = cv2.cvtColor(leftMiddleImg,cv2.COLOR_RGB2GRAY)
rightMiddleImg = cv2.cvtColor(rightMiddleImg,cv2.COLOR_RGB2GRAY)
#leftMiddleImg = cv2.flip(rightMiddleImg,1)
    
# resize(leftImg,50)
# resize(rightImg,50)
rightImgBig = resize(rightImgBig,145)
leftImgBig = resize(leftImgBig,145)
# resize(rightMiddleImg,50)

start_time = timeit.default_timer()

def checkNormalHit():
    sct = mss()
    im_bgr = np.array(sct.grab(monitor = {'top': 720, 'left': 400, 'width': 1100, 'height': 300}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2GRAY)  # 2
    
    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # left
    result = cv2.matchTemplate(im_bgr,leftImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # print("left={}".format(max_val))
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
    # im_bgr = cv2.rectangle(im_bgr,max_loc,(max_loc[0]+rightImgBig.shape[1],max_loc[1]+rightImgBig.shape[0]),(255,255,255),2)
    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # print("right={}".format(max_val))
    if max_val > decisionThreshold:
        pyautogui.press('right')
        return True
    result = cv2.matchTemplate(im_bgr,leftImgBig,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # im_bgr = cv2.rectangle(im_bgr,max_loc,(max_loc[0]+leftImgBig.shape[1],max_loc[1]+leftImgBig.shape[0]),(255,255,255),2)
    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # print(leftImgBig.shape)
    # print("right={}".format(max_val))

    if max_val > decisionThreshold:
        pyautogui.press('left')
        return True

    return False
    
def checkMiddleHit():
    sct = mss()
    im_bgr = np.array(sct.grab(monitor = {'top': 170, 'left': 750, 'width': 360, 'height': 310}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2GRAY)  # 2



    resize(im_bgr,50)
    # left
    result = cv2.matchTemplate(im_bgr,leftMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result)

    # right
    result = cv2.matchTemplate(im_bgr,rightMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result)
    im_bgr = cv2.rectangle(im_bgr,max_loc_right,(max_loc_right[0]+rightMiddleImg.shape[1],max_loc_right[1]+rightMiddleImg.shape[0]),(255,0,0),2)
    cv2.imshow('window',im_bgr)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # print("left={}, right={}".format(max_val_left,max_val_right))

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