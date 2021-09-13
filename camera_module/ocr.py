#!/usr/bin/env python
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2 as cv
import os
import numpy as np

def run_ocr(image, thresh_preprocess, blur_preprocess):
    # load the example image and convert it to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    if thresh_preprocess:
        gray = cv.threshold(gray, 0, 255,
            cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    if blur_preprocess:
        gray = cv.medianBlur(gray, 3)

    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv.erode(gray, kernel, iterations=2)
    gray = cv.dilate(img_erosion, kernel, iterations=2)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print(text)

    # show the output images
    cv.imshow("Image", image)
    cv.imshow("Output", gray)
    # cv.waitKey(0)
