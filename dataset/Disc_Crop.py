from PIL import Image
import numpy as np
import os
from scipy import misc
import argparse
import sys

'''
function: crop the union Disc area of two eyes and switch it
          not mix Glaucoma and Non-Glaucoma together
          for 400 images it crops 200 times
label_file_dir: GT images
               structure:-Disc_Cup_Masks
                         -----Glaucoma
                         ---------g0001.png to g0040.png
                         -----Non-Glaucoma
                         ---------n0001.png to n0360.png
origin_file_dir: Ophthalmology photos
               structure:-Training400
                         -----Training400
                         ---------Glaucoma
                         -------------g0001.png to g0040.png
                         ---------Non-Glaucoma
                         -------------n0001.png to n0360.png
label_save_dir: cropped GT images
origin_save_dir: cropped Ophthalmology photos
'''

def crop(label_file_dir, origin_file_dir, label_save_dir, origin_save_dir):
    dir_list = []

    for root, dirs, files in os.walk(label_file_dir):
        dir_list = dirs
        break
    for i in dir_list:
        for root, dirs, files in os.walk(label_file_dir + '/' + i):
            for i1 in range(0, len(files), 2):
                min_row = 2056
                max_row = 0
                min_col = 2124
                max_col = 0
                image1 = Image.open(label_file_dir + '/' + i + '/' + files[i1])
                image2 = Image.open(label_file_dir + '/' + i + '/' + files[i1 + 1])
                image_matrix1 = np.array(image1)
                image_matrix2 = np.array(image2)
                rows, cols= image_matrix1.shape
                for j in range(0, rows):
                    for k in range(0, cols):
                        if (image_matrix1[j, k] != 0 and image_matrix1[j, k] != 255) or (image_matrix2[j, k] != 0 and image_matrix2[j, k]) != 255:
                            if j < min_row: min_row = j
                            if j > max_row: max_row = j
                            if k < min_col: min_col = k
                            if k > max_col: max_col = k
                image3 = Image.open(origin_file_dir + '/' + i + '/' + files[i1])
                image4 = Image.open(origin_file_dir + '/' + i + '/' + files[i1 + 1])
                image_matrix3 = np.array(image3)
                image_matrix4 = np.array(image4)
                rows, cols, deep = image_matrix3.shape
                for j in range(min_row, max_row + 1):
            for k in range(min_col, max_col + 1):
                        image_matrix1[j, k], image_matrix2[j, k] = image_matrix2[j, k], image_matrix1[j, k]
                        for l in range(deep):
                            image_matrix3[j, k, l], image_matrix4[j, k, l] = image_matrix4[j, k, l], image_matrix3[j, k, l]
                misc.imsave(label_save_dir + '/' + i + '/' + files[i1], image_matrix1)
                misc.imsave(label_save_dir + '/' + i + '/' + files[i1 + 1], image_matrix2)
                misc.imsave(origin_save_dir + '/' + i + '/' + files[i1], image_matrix3)
                misc.imsave(origin_save_dir + '/' + i + '/' + files[i1 + 1], image_matrix4)
            break



if __name__ == '__main__':

    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_file_dir", type=str)
    parser.add_argument("--origin_file_dir", type=str)
    parser.add_argument("--label_save_dir", type=str)
    parser.add_argument("--origin_save_dir", type=str)
    args = parser.parse_args()
    # call the "main" function
    crop(args.label_file_dir, args.origin_file_dir, args.label_save_dir, args.origin_save_dir)
