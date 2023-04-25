
#!/usr/bin/env python3

"""
Title: Assignment 03 - Image Descriptors
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429
"""

import sys
import imageio.v2 as imageio


def read_input():
    """
    Reads input from stdin.

    Returns:
        X0 (list): images without humans
        X1 (list): images with humans
        Xtest (list): images to be classified 
    """
    # Read input from stdin line by line, remove trailing whitespace and split each line into a list of words
    lines = [line.rstrip().split() for line in sys.stdin]

    # Read images from files
    files = [imageio.imread(file) for line in lines for file in line]

    # Split images into X0, X1 and Xtest
    X0 = files[0:10]
    X1 = files[10:20]
    Xtest = files[20:30]

    return X0, X1, Xtest


def main():
    X0, X1, Xtest = read_input()


if __name__ == '__main__':
    main()
