import os

directory_path = 'Photos/'
for image in os.listdir(directory_path):
    file_image = open(os.path.join(directory_path, image))
    print(file_image.name)