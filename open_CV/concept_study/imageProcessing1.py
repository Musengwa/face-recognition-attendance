import cv2 as cv

#read the image
img = cv.imread('myImage.png')

# convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#applygussianBlur
blurred = cv.GaussianBlur(gray, (5,5), 5)

#detect edges using canny
edge = cv.Canny (gray, 150, 150)

#display all results
cv.imshow('original', img)
cv.imshow('grayscale', gray)
cv.imshow('blur', blurred)
cv.imshow('Edges', edge)

# save img
cv.imwrite('output.jpg', blurred)
cv.waitKey(0)
cv.destroyAllWindows()
