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

# Resize image by the default dimensions, adding padding and cropping if it's necessary
def resizeWithPadding(img, ratio, default_eye1, eye1, default_dimensions):
    img = cv.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))
    
    # Add padding if image is minor than the default one
    if(img.shape[1] < default_dimensions[0]):
        img = cv.copyMakeBorder(img, 0, 0, 0, default_dimensions[0] - img.shape[1], cv.BORDER_CONSTANT)        
    if(img.shape[0] < default_dimensions[1]):
        img = cv.copyMakeBorder(img, 0, default_dimensions[1] - img.shape[0], 0, 0, cv.BORDER_CONSTANT)
    
    # Translate image to default eyes positions
    x_eye_resized = int(eye1[0] * ratio)
    y_eye_resized = int(eye1[1] * ratio)
    img = translate(img, default_eye1[0] - x_eye_resized, default_eye1[1] - y_eye_resized)
    
    # Crop image if it's necessary
    if(img.shape[1] > default_dimensions[0]):
        img = img[:,:default_dimensions[0]]
    if(img.shape[0] > default_dimensions[1]):
        img = img[:default_dimensions[1],:]

    return img 

# ----------- Main program ----------- 
# Get script arguments
input_folder = ''
output_folder = ''
if __name__ == "__main__":
    getScriptArguments(sys.argv)

# Program variables
loop_enable = 1
loop_delimiter = 375
default_eye1 = (0,0)
default_eye2 = (0,0)
default_dimensions = (0,0)

# Open folder and list files of directory
for count, img_name in enumerate(sorted(os.listdir(input_folder)), 1):
    if(loop_enable == 0) and (loop_delimiter != count):
        continue
    if(count > loop_delimiter):
        break
    
    img_file = open(os.path.join(input_folder, img_name))

    # Reset eyes position
    eye1 = (0,0)
    eye2 = (0,0)
    
    # Load image and resize for debugging
    img = cv.imread(img_file.name)
    
    # Convert image to gray scale for eyes detection
    #img = cv.resize(img, (img.shape[1]//5, img.shape[0]//5))    # Resize image for computing performance
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Eyes detection using haarcascades classifier
    eye_cascade = cv.CascadeClassifier(os.path.join(cv.data.haarcascades, 'haarcascade_eye.xml'))
    eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
    for i, (ex, ey, ew, eh) in enumerate(eyes):
        cv.rectangle(img, (ex,ey), (ex + ew, ey + eh), (0,255,0))
        cv.circle(img, (ex + ew//2, ey+eh//2), 1, (0,255,0), cv.FILLED)
        if(i == 0):
            eye1 = (ex + ew//2, ey+eh//2)
        else:
            eye2 = (ex + ew//2, ey+eh//2)
            break
        
    # If an eye isn't detected
    print(eye1, eye2)
    if(eye1 == (0,0)) or (eye2 == (0,0)):
        print(f'ERROR! {img_name} had a problem in eye detection!')
        continue    
    
    # Case eye2 is the left one -> invert
    if(eye1[0] > eye2[0]):
        eye1, eye2 = eye2, eye1
        
    # Math to find the angle between eye's positions
    eye_hip = math.sqrt(math.pow((eye1[0] - eye2[0]), 2) + math.pow((eye1[1] - eye2[1]), 2))
    eye_tan = (eye2[1] - eye1[1]) / (eye2[0] - eye1[0])
    degree = math.degrees(math.atan(eye_tan))
    img = rotate(img, degree, (int(eye2[0]), int(eye2[1])))

    # Set new eye1 position
    eye1 = (int(eye2[0] - eye_hip), eye2[1])
    
    # Draw a line to see if rotate was sucessful -> using photo eyes
    # cv.line(img, (0, eye1[1]), (img.shape[1], eye1[1]), (0,255,0))
    # cv.line(img, (eye2[0], 0), (eye2[0], img.shape[0]), (0,255,0))
    # cv.line(img, (eye1[0], 0), (eye1[0], img.shape[0]), (0,255,0))
    
    # Set default eyes
    if(default_eye1 == (0,0)):
        default_eye1 = (eye1[0], eye1[1])
        default_eye2 = (eye2[0], eye2[1])
        default_eyes_distance = eye2[0] - eye1[0]
        default_dimensions = (img.shape[1], img.shape[0])
    
    if(count > 1) and (default_eye1 != (0,0)):    
        # Resize image to the default image eyes distant ratio - CORRIGIR!!!!
        ratio = default_eyes_distance / (eye2[0] - eye1[0])
        if(ratio <= 0):
            print(f'ERROR! {img_name} had a problem in ratio calculus!')
            print(f'Ratio = {default_eyes_distance} / ({eye2[0]} - {eye1[0]})')
            continue
        img = resizeWithPadding(img, ratio, default_eye1, eye1, default_dimensions)
    
    # Draw a line to see if rotate was sucessful -> using default eyes
    cv.line(img, (0, default_eye1[1]), (img.shape[1], default_eye1[1]), (0,255,0))
    cv.line(img, (default_eye1[0], 0), (default_eye1[0], img.shape[0]), (0,255,0))
    cv.line(img, (default_eye2[0], 0), (default_eye2[0], img.shape[0]), (0,255,0))
        
    # cv.imshow(f'Image{count}', img)
    cv.imwrite(f'{output_folder}/photo{count}.jpg', img)       
    print(f'{img_name} successfully normalizated!\n\n')

cv.waitKey(0)