import cv2
import sys
from random import randint

TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_SOURCE = 'videos/people.mp4'
BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']
BGS_TYPE = BGS_TYPES[4]
#gmg = 24
#mog2 = 8
#mog = 14
#knn = 8
#cnt = 6
def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=10)
    if BGS_TYPE == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Unknown createBackgroundSubtractor type")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)
bg_subtractor = getBGSubtractor(BGS_TYPE)
e1 = cv2.getTickCount()

def main():
    frame_number = -1
    while (cap.isOpened):
        ok, frame = cap.read()
        # print(ok)

        if not ok:
            print('finish processing the video')
            break

        frame_number += 1

        bg_mask = bg_subtractor.apply(frame)
        res = cv2.bitwise_and(frame, frame, mask=bg_mask)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', res)

        if cv2.waitKey(1) & 0xFF == ord("q") or frame_number > 250:
            break

    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    print(t)

main()

