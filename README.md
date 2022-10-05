# Face sobreposition normalization

## Description

The main goal of this project is to create an application to reposition a set of photos in a default face position,
creating a sobreposition of faces. For that, it was used Python language with OpenCV for image transformation.

## How to Use

After downloading the project, it's necessary to create an input folder, where all the images to be normalizated are, and an output folder, where
the normalizated images are going. The running process is described below.

Arguments:

`-i` or `--input-folder` is the path to the folder that contains all the images to be normalizated

`-o` or `--output-folder` is the path where all the images normalizated are going

`-h` or `--help` print a basic guide in terminal

`-s` or `--size-ratio` is a value used to resize the images by multiplying it's width and height

Basic command `python face-sobreposition-normalization -i <input_folder> -o <output_folder>`
