#!/usr/bin/env python3

"""
Title: Assignment 04 - Mathematical morphology
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429
"""

import numpy as np
import imageio.v2 as imageio


def read_input():
    """
    Reads input from stdin and loads image.

    Returns:
        image: binary image loaded from provided filename
        i_k: seed pixel x coordinate
        j_k: seed pixel y coordinate
        c: connectivity in inclusive range [4, 8]
    """
    # Read first line, convert to string and remove trailing newline
    filename = str(input().rstrip())

    # Read second line, convert to integer and remove trailing newline
    i_k = int(input().rstrip())

    # Read third line, convert to integer and remove trailing newline
    j_k = int(input().rstrip())

    # Read fourth line, convert to integer and remove trailing newline
    c = int(input().rstrip())

    # Load image as binary, avoiding compression artifacts, by converting it to boolean and back
    image = (imageio.imread(filename) > 127).astype(np.uint8)

    return image, i_k, j_k, c


def main():
    """
    Program entry point.
    """
    image, i_k, j_k, c = read_input()


if __name__ == '__main__':
    main()
