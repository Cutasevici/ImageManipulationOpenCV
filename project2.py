import numpy as np
import cv2
import sys

TEXT_COLOR = (24, 201, 255)
TRACKER_COLOR = (255, 128, 0)
WARNING_COLOR = (24, 201, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/pedestrians.mp4'

BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
BGS_TYPE = BGS_TYPES[0]

def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3,5), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((11,11), np.uint8)
    return kernel

def getFilter(img,filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                getKernel("closing"),
                                iterations=2)
    if filter == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN,
                                getKernel("opening"), iterations=2)

    if filter== "dilation":
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == "combine":
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                   getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN,
                                   getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print('Invalid detector!')
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = getBGSubtractor(BGS_TYPE)