import cv2 as cv
import numpy as np

cap = cv.VideoCapture(-1)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv.imshow('Frame',frame)

        boundaries = [([17, 15, 100], [50, 56, 200])]
        # loop over the boundaries
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            # find the colors within the specified boundaries and apply the mask
            
            mask = cv.inRange(frame, lower, upper)
            output = cv.bitwise_and(frame, frame, mask = mask)

            # get contours on output of mask
            _, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            max_area = 0
            for contour in contours:
                area = cv.contourArea(contour)
                if area > max_area: max_area = area
            
            if max_area > 1000:
                print("I see red :)")

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv.destroyAllWindows()
