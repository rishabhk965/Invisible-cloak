import numpy as np;
import cv2          # computer vision
import time         # To let the camera take its time to set-up
cap = cv2.VideoCapture(0)       # create a video-capture object 

time.sleep(2)               # time for the camera to start

background = 0

for i in range(100):
    ret,background = cap.read()           # to read the background (50 times is to read the background many times and take the best capture)

while(cap.isOpened()):
    ret,img = cap.read()            # to read the current image
    if not ret:             # ret returns true until the capturing is interrupted
        break

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)   # HSB (BGR is what our cam captures)   Converted the BGR image into hsv

    lower_color = np.array([0,120,70])      # numpy arrays with arguments as [hue ,saturation ,brightness ]
    upper_color = np.array([10,255,255])

    mask1 = cv2.inRange(hsv,lower_color,upper_color)        # separating the cloak part

    lower_color = np.array([170,120,70])      # numpy arrays with arguments as [hue ,saturation ,brightness ]
    upper_color = np.array([180,255,255])

    mask2 = cv2.inRange(hsv,lower_color,upper_color)        # separating the cloak part

    mask1 = mask1 + mask2       # OR 1 or x  Operator overloading

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=2)      # Noise Removal using Morphology function
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)      # iterations are for smoothing the image

    mask2 = cv2.bitwise_not(mask1)          # mask2 is is everything except the cloak

    rest1 = cv2.bitwise_and(background,background, mask=mask1)      # used for segmentation of color
    rest2 = cv2.bitwise_and(img,img, mask=mask2)       # used to substitute the cloak part
    # background is the image captucolor before cloak and img is the image captucolor after so we just replacing the cloak part using background image
    # (Super imposing 2 images)
    final_output = cv2.addWeighted(rest1,1,rest2,1,0)           # adding two images [ res1*alpha(1) + res2*beta(1) + gamma(0) ]

    cv2.imshow('Hola',final_output)     # display final_output
    k = cv2.waitKey(10)
    if k==27:                   # so that when Esc is pressed the program quits executing
        break

cap.release()
cv2.destroyAllWindows()

    # Change Hue Value in Line 20,21,25,26 as per cloak color