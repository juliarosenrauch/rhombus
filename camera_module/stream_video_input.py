#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from ocr import run_ocr

# -1 for pi
# 0 for laptop webcam

cap = cv.VideoCapture(1)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

outer_count = 0
# Read until video is completed
while (cap.isOpened() and not (cv.waitKey(25) & 0xFF == ord('q'))):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_for_ocr = frame.copy()

    if ret == True:
        # BGR order
        boundaries = [([0, 0, 130], [70, 70, 255])]
        # loop over the boundaries
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            # find the colors within the specified boundaries and apply the mask
            mask = cv.inRange(frame, lower, upper)
            #output = cv.bitwise_and(frame, frame, mask = mask)

            # get contours on output of mask
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_area = 0
            index_of_largest = -1
            index = 0
            largest_contour = []
            for contour in contours:
                area = cv.contourArea(contour)
                if area > max_area: 
                    max_area = area
                    index_of_largest = index
                    largest_contour = contour
                index += 1
            
            if max_area > 4000:
                cv.drawContours(frame, contours, index_of_largest, (0,0,255), 3)
                x,y,w,h = cv.boundingRect(largest_contour)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                frame_for_ocr = frame_for_ocr[y:y+h, x:x+w]
                if outer_count%10 == 0:
                    run_ocr(frame_for_ocr, True, True)

        cv.imshow('Frame',frame)
        outer_count += 1

    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv.destroyAllWindows()
