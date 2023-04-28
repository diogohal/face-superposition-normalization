# Face superposition normalization

## Description

The main goal of this project is to create an application to reposition a set of photos in a default face position, creating a sobreposition of faces. For that, it was used Python language with OpenCV for image transformation and Haar Cascade classifier for eyes detection.

## How to Use

After downloading the project, it's necessary to create an input folder, where all the images to be normalized are, and an output folder, where
the normalized images are going. For testing purpose, the photos inside test_samples folder can be used. The running process is described below.

Arguments:

`-h` or `--help` print a basic guide in terminal

`-i` or `--input-folder` is the path to the folder that contains all the images to be normalized

`-o` or `--output-folder` is the path to the folder where all the images normalized are going

`-d` or `--default-image` is the image filename inside input folder which it will be used to set default face position. Default: first image to be normalized

`-q` or `--quantity` is the quantity of images to be normalized. Default: all images inside input folder

`-s` or `--size-ratio` is a value used to resize the images by multiplying it's width and height. Default: 1.0

`-l` or `--lines` draw horizontal and vertical lines with intersections where default eyes position are

Basic command `python face-sobreposition-normalization.py -i <input_folder> -o <output_folder>`

## Possible problems

The Haar Cascade classifier sometimes has problems in eyes recognition, such as incorrect detection or lack of recognition. Upside down images also can't be inverted and are normalized incorrectly.
