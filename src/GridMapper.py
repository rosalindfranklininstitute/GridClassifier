'''
Created on 23 Mar 2020

@author: Mark Basham
'''
import os
from matplotlib.pyplot import imread, imsave

def process_grid(grid_image_filename, output_directory):
    grid_image = imread(grid_image_filename)

    # TODO replace these features with real ones.
    feature1 = grid_image.mean(2)
    feature2 = feature1*2.0
    feature2[feature1>feature1.mean()] = 0

    # Save the features to the output directory
    imsave(os.path.join(output_directory,'feature1.png'), feature1)
    imsave(os.path.join(output_directory,'feature2.png'), feature2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_image", help="The image of the grid you wish to process")
    parser.add_argument("output_dir", help="The output directory to write the resulting images to")
    args = parser.parse_args()
    process_grid(args.grid_image, args.output_dir)
