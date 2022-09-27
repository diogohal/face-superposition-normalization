from operator import mod
import cv2 as cv
import math
import os
import numpy as np

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    dimensions = (width, height)
    
    if(rotPoint == None):
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    
    return cv.warpAffine(img, rotMat, dimensions)


default_eyes_position = ((0,0), (0,0))
# Open folder and list files of directory
directory_path = 'Photos/'
for count, image in enumerate(os.listdir(directory_path)):
    img_file = open(os.path.join(directory_path, image))

    # Reset eyes position
    eye2 = (0,0)
    eye1 = (0,0)
    
    # Load image and resize for debugging
    img = cv.imread(img_file.name)
    img = cv.resize(img, (img.shape[1]//5, img.shape[0]//5))
    img_size = (img.shape[1], img.shape[0])

    # Eyes detection using haarcascades classifier
    eye_cascade = cv.CascadeClassifier(os.path.join(cv.data.haarcascades, 'haarcascade_eye.xml'))
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        cv.rectangle(img, (ex,ey), (ex + ew, ey + eh), (0,255,0))
        cv.circle(img, (ex + ew//2, ey+eh//2), 1, (0,255,0), cv.FILLED)
        if i == 0:
            eye1 = (ex + ew//2, ey+eh//2)
        else:
            eye2 = (ex + ew//2, ey+eh//2)
    
    # If an eye isn't detected
    if(eye1 == (0,0)) or (eye2 == (0,0)):
        continue 
    
    # Math to find the angle between eye's positions
    eye_hip = math.sqrt(math.pow((eye1[0] - eye2[0]), 2) + math.pow((eye1[1] - eye2[1]), 2))
    eye_tan = (eye2[1] - eye1[1]) / (eye2[0] - eye1[0])
    degree = math.degrees(math.atan(eye_tan))
    img = rotate(img, degree, (int(eye2[0]), int(eye2[1])))

    # Set new eye1 position
    eye1 = (int(eye2[0] - eye_hip), eye2[1])
    
    # Draw a line to see if rotate was sucessful
    cv.line(img, (0, eye2[1]), (img.shape[1], eye2[1]), (0,255,0))
    cv.line(img, (0, eye2[1]), (img.shape[1], eye2[1]), (0,255,0))
    # cv.imshow('Image', img)   
    
    # Set default eyes position
    if(default_eyes_position == ((0,0),(0,0))):
        default_eyes_position = (eye1, eye2)
        
    # EVERYTHING IS BROKEN BELOW
    else:
        # Resize image
        coef = (eye2[0] - eye1[0]) / (default_eyes_position[1][0] - default_eyes_position[0][0])
        img = cv.resize(img, (int(img_size[0]*coef), int(img_size[1]*coef)), cv.INTER_AREA)
        img = img[:img_size[1], :img_size[1]]
        
    # Translate image to default eyes position
    img = translate(img, default_eyes_position[0][0] - eye1[0], default_eyes_position[0][1] - eye1[1])
    cv.line(img, (eye1[0], 0), (eye1[0], img.shape[0]), (0,0,255), 1)
    cv.line(img, (eye2[0], 0), (eye2[0], img.shape[0]), (0,0,255), 1)
        
    cv.imwrite(f'Photos_out/img_{count}.jpg', img)
    #cv.imshow(f'Rotate{count}', img)

cv.waitKey(0)