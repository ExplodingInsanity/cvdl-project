import sys
import pyautogui
import cv2 #pip install opencv-contrib-python
import numpy as np
from time import sleep
from PIL import ImageGrab
import timeit

decisionThreshold = .8

def resize(img,scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim)

leftImg = cv2.imread('images/left_small.png');
rightImg = cv2.imread('images/right_small.png');

resize(leftImg,50)
resize(rightImg,50)

start_time = timeit.default_timer()

def checkNormalHit():
    pass
    

while(True):
    # start_time = timeit.default_timer()
    printscreen_pil =  ImageGrab.grab(bbox=(600, 700, 1320, 800))
    # elapsed = timeit.default_timer() - start_time
    # print("screenshot - ",elapsed)
    # start_time = timeit.default_timer()
    printscreen_numpy =   np.array(printscreen_pil,dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
    # elapsed = timeit.default_timer() - start_time
    # print("numpy - ",elapsed)
    # start_time = timeit.default_timer()
    im_bgr = cv2.cvtColor(printscreen_numpy, cv2.COLOR_RGB2BGR)
    # im_bgr = resize(im_bgr,50)
    # elapsed = timeit.default_timer() - start_time
    # print("bgr conversion - ",elapsed)
    # start_time = timeit.default_timer()

    # im_bgr = cv2.resize(im_bgr,(1280,720), interpolation = cv2.INTER_AREA)
    # left
    result = cv2.matchTemplate(im_bgr,leftImg,cv2.TM_CCOEFF_NORMED)
    # elapsed = timeit.default_timer() - start_time
    # print("match template - ",elapsed)
    # start_time = timeit.default_timer()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # elapsed = timeit.default_timer() - start_time
    # print("minmaxloc - ",elapsed)

    w = leftImg.shape[1]
    h = leftImg.shape[0]
    if max_val > decisionThreshold:
        # print("left",str(max_val))
        pyautogui.press('left')
        # cv2.rectangle(im_bgr,max_loc,(max_loc[0]+w,max_loc[1]+h),(0,255,255),2)
        continue

    # right
    result = cv2.matchTemplate(im_bgr,rightImg,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    w = rightImg.shape[1]
    h = rightImg.shape[0]
    if max_val > decisionThreshold:
        # print("right",str(max_val))
        pyautogui.press('right')
        # cv2.rectangle(im_bgr,max_loc,(max_loc[0]+w,max_loc[1]+h),(0,255,255),2)
        continue

    # cv2.putText(im_bgr,str(max_val),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_AA)

    # cv2.imshow('window',im_bgr)
    # if cv2.waitKey(20) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break