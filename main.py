import pyautogui
import cv2 #pip install opencv-contrib-python
import numpy as np
from time import sleep
from mss import mss
import timeit
from PIL import Image

decisionThreshold = .8
decisionThresholdCenter = .6

def resize(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

leftImg = cv2.imread('images/left_small.png');
rightImg = cv2.imread('images/right_small.png');
leftMiddleImg = cv2.imread('images/left_center_tip.png');
rightMiddleImg = cv2.imread('images/right_center_tip.png');

resize(leftImg,50)
resize(rightImg,50)
resize(leftMiddleImg,50)
resize(rightMiddleImg,50)

start_time = timeit.default_timer()

def checkNormalHit():
    sct = mss()
    im_bgr = np.array(sct.grab(monitor = {'top': 720, 'left': 600, 'width': 750, 'height': 200}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2BGR)  # 2
    # left
    result = cv2.matchTemplate(im_bgr,leftImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    w = leftImg.shape[1]
    h = leftImg.shape[0]
    if max_val > decisionThreshold:
        pyautogui.press('left')
        return True
    # right
    result = cv2.matchTemplate(im_bgr,rightImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    w = rightImg.shape[1]
    h = rightImg.shape[0]
    if max_val > decisionThreshold:
        pyautogui.press('right')
        return True
    return False
    
def checkMiddleHit():
    sct = mss()
    im_bgr = np.array(sct.grab(monitor = {'top': 170, 'left': 800, 'width': 250, 'height': 310}))
    im_bgr = np.flip(im_bgr[:, :, :3], 2)  # 1
    im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_RGB2BGR)  # 2

    resize(im_bgr,50)
    # left
    result = cv2.matchTemplate(im_bgr,leftMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_left, max_val_left, min_loc_left, max_loc_left = cv2.minMaxLoc(result)

    # right
    result = cv2.matchTemplate(im_bgr,rightMiddleImg,cv2.TM_CCOEFF_NORMED)
    min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(result)

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