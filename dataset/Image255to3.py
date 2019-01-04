from PIL import Image
import numpy as np
import os
from scipy import misc
import argparse
import sys
'''
function: change GT image label color for training
          white(255) -> 0
          black(0) -> 2
          grey(1-254) -> 1
file_dir: folder which contains all GT image(g0001.png-g0040.png, n0001.png-n0360.png)
save_dir: changed result
'''

def func(file_dir, save_dir):
    for root, dirs, files in os.walk(file_dir):
        for i in files:
            image = Image.open(root+'/'+i)
            image_matrix = np.array(image)
            rows, cols= image_matrix.shape
            for j in range(0, rows):
                for k in range(0, cols):
                    #white(255) to 0
                    if image_matrix[j, k] == 255:
                        image_matrix[j, k] = 0
                    else:
                        #black(0) to 2
                        if image_matrix[j, k] == 0:
                            image_matrix[j, k] = 2
                        else: 
                            #grey to 1
                            image_matrix[j, k] = 1
                       
            misc.imsave(save_dir + '/' + i, image_matrix)

if __name__ == '__main__':
    # create an argument parser to control the input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()
    # call the "main" function
    func(args.file_dir, args.save_dir)
