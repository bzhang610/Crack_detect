import os
from utils import *

if __name__ == "__main__":
    '''
    This is how we handle loading the input dataset, running your function, and printing the output
    '''
    from sys import argv
    img_file = argv[1]                  # get file name from input
    detect_crack(img_file)              # run program
