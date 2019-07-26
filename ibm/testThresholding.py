import imutils
import cv2
 

image = cv2.imread("img_1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("frame", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()