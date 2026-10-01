import cv2 as cv

# Initialize detector and recognizer
detector = cv.FaceDetectorYN.create(
    'models/face_detection_yunet_2026may.onnx',
    "", (320, 320), 0.9, 0.3, 5000
)

recognizer = cv.FaceRecognizerSF.create(
    'models/face_recognition_sface_2021dec.onnx', ""
)

# Load two images
img1 = cv.imread('tests/images/person3a.jpeg')
img2 = cv.imread('tests/images/person2b.jpeg')

# Detect faces
detector.setInputSize((img1.shape[1], img1.shape[0]))
faces1 = detector.detect(img1)

detector.setInputSize((img2.shape[1], img2.shape[0]))
faces2 = detector.detect(img2)

if faces1[1] is None or faces2[1] is None:
    print("No face detected")
else:
    # Align and extract features
    face1_align = recognizer.alignCrop(img1, faces1[1][0])
    face2_align = recognizer.alignCrop(img2, faces2[1][0])
    
    face1_feature = recognizer.feature(face1_align)
    face2_feature = recognizer.feature(face2_align)
    
    # Compare faces
    cosine_score = recognizer.match(
        face1_feature, face2_feature, 
        cv.FaceRecognizerSF_FR_COSINE
    )
    
    l2_score = recognizer.match(
        face1_feature, face2_feature, 
        cv.FaceRecognizerSF_FR_NORM_L2
    )
    
    # Thresholds
    cosine_threshold = 0.363
    l2_threshold = 1.128
    
    if cosine_score >= cosine_threshold:
        print(f"Same person (Cosine: {cosine_score:.3f})")
    else:
        print(f"Different person (Cosine: {cosine_score:.3f})")
    
    if l2_score <= l2_threshold:
        print(f"Same person (L2: {l2_score:.3f})")
    else:
        print(f"Different person (L2: {l2_score:.3f})")