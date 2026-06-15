import cv2

#read image
image = cv2.imread("myImage.png")

#display image
cv2.imshow('My Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
