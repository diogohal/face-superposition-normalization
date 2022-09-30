import cv2 as cv
import math
import os
import numpy as np
import sys
import getopt

# ----------- Transformation functions ----------- 
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

def getScriptArguments(argv):
    global input_folder
    global output_folder
    arg_help = "{0} -i <input_folder> -o <output_folder>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "h:i:o:", ["help", "input=", "output="])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-i", "--input"):
            input_folder = arg
        elif opt in ("-o", "--output"):
            output_folder = arg

    print('input:', input_folder)
    print('output:', output_folder)

# ----------- Main program ----------- 
# Get script arguments
input_folder = ''
output_folder = ''
if __name__ == "__main__":
    getScriptArguments(sys.argv)

# Program variables
loop_enable = 1
loop_delimiter = 176
default_eyes_height = 0
default_eyes_widht = 0

# Open folder and list files of directory
for count, image in enumerate(sorted(os.listdir(input_folder)), 1):
    if(loop_enable == 0) and (loop_delimiter != count):
        continue
    if(count > loop_delimiter):
        break
    
    img_file = open(os.path.join(input_folder, image))

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
        if(i == 0):
            eye1 = (ex + ew//2, ey+eh//2)
            continue
        # Case eye2 is the right one
        if((ex - eye1[0]) > 0):
            eye2 = (ex + ew//2, ey+eh//2)
            break
        # Case eye2 is the left one -> invert
        else:
            eye2 = eye1
            eye1 = (ex + ew//2, ey+eh//2)
            break
    
    # If an eye isn't detected
    if(eye1 == (0,0)) or (eye2 == (0,0)):
        print(f'Photo {count} had a problem in eye detection!')
        cv.imwrite(f'errors/photo{count}_error.jpg', img)
        continue    
    
    # Math to find the angle between eye's positions
    eye_hip = math.sqrt(math.pow((eye1[0] - eye2[0]), 2) + math.pow((eye1[1] - eye2[1]), 2))
    eye_tan = (eye2[1] - eye1[1]) / (eye2[0] - eye1[0])
    degree = math.degrees(math.atan(eye_tan))
    img = rotate(img, degree, (int(eye2[0]), int(eye2[1])))

    # Set new eye1 position
    if(eye1[0] < eye2[0]):
        eye1 = (int(eye2[0] - eye_hip), eye2[1])
    else:
        eye2 = (int(eye1[0] - eye_hip), eye1[1])
    
    # Draw a line to see if rotate was sucessful
    cv.line(img, (0, eye1[1]), (img.shape[1], eye1[1]), (0,255,0))
    cv.line(img, (eye2[0], 0), (eye2[0], img.shape[0]), (0,255,0))
    cv.line(img, (eye1[0], 0), (eye1[0], img.shape[0]), (0,255,0))
    
    # Set default eyes height
    if(default_eyes_height == 0):
        default_eyes_height = eye1[1]
        default_eyes_widht = eye1[0]
        default_eyes_distance = eye2[0] - eye1[0] 
    
    # Translate image to the default eyes height and width positions
    if(count > 0):
        img = translate(img, 0, default_eyes_height - eye2[1])
        img = translate(img, default_eyes_widht - eye1[0], 0)
    
    # Resize image to the default image eyes distant ratio
    ratio = default_eyes_distance / (eye2[0] - eye1[0])
    cv.resize(img, (int(img.shape[1] * ratio), int(img.shape[0]*ratio)))
    
    # cv.imshow(f'Image{count}', img)
    cv.imwrite(f'{output_folder}/photo{count}.jpg', img)       
    print(f'Photo {count} successfully normalizated!')

cv.waitKey(0)