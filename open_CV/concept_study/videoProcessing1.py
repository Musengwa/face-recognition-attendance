import cv2 as cv


#open webcam (0 = default camera)
cap = cv.VideoCapture(0)

#OR OPEN A VIDEO FILE 
#cap = cv.videocapture('video.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #display
    cv.imshow('webcam', gray)

    #break on 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

