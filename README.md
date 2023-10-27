# Face superposition normalization

## Description

The main goal of this project is to create an application to reposition a set of photos to a default face position, creating a superposition of faces. For that, Python language with OpenCV for image transformation and Haar Cascade classifier for eye detection were used.

## How to Use

After downloading the project, you need to create an input folder where all the images to be normalized are stored and an output folder where the normalized images will be saved. For testing purposes, you can use the photos inside the "test_samples" folder. The running process is described below.

Arguments:

`-h` or `--help`: Print a basic guide in the terminal.

`-i` or `--input-folder`: The path to the folder that contains all the images to be normalized.

`-o` or `--output-folder`: The path to the folder where all the normalized images will be saved.

`-d` or `--default-image`: The filename of the image inside the input folder that will be used to set the default face position. Default: the first image to be normalized.

`-q` or `--quantity`: The quantity of images to be normalized. Default: all images inside the input folder.

`-s` or `--size-ratio`: A value used to resize the images by multiplying their width and height. Default: 1.0.

`-l` or `--lines`: Draw horizontal and vertical lines with intersections where the default eye positions are.

Basic command `python face-sobreposition-normalization.py -i <input_folder> -o <output_folder>`

## Possible problems

The Haar Cascade classifier sometimes has problems with eye recognition, such as incorrect detection or a lack of recognition. Upside-down images also cannot be inverted and are normalized incorrectly.
