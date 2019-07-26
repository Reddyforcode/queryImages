import cv2
import numpy as np

img = cv2.imread('img_1.jpg')

ORANGE_MIN = np.array([5, 50, 50],np.uint8)
ORANGE_MAX = np.array([15, 255, 255],np.uint8)

low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)


frame2 = cv2.inRange(hsv_img, low_red, high_red)
cv2.imwrite('output1.jpg', frame_threshed)
cv2.imwrite('output3.jpg', frame2)