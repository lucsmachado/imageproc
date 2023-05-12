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


class Pixel:
    """
    Class representing a pixel in an image.
    """

    def __init__(self, x, y):
        """
        Constructor.

        Args:
            x: x coordinate of pixel
            y: y coordinate of pixel
        """
        self.x = x
        self.y = y

    def __eq__(self, other):
        """
        Override equal operator.

        Args:
            other: some other Pixel instance for comparison

        Returns:
            True if self is equal to other, False otherwise
        """
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        """
        Override not equal operator.

        Args:
            other: some other Pixel instance for comparison

        Returns:
            True if self is not equal to other, False otherwise
        """
        return self.x != other.x or self.y != other.y

    def __lt__(self, other):
        """
        Override less than operator.

        Args:
            other: some other Pixel instance for comparison

        Returns:
            True if self is less than other, False otherwise
        """
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x


def flood_fill_binary(image, seed_x, seed_y, c):
    """
    Finds connected components in a binary image using flood fill algorithm.

    Args:
        image: binary image matrix
        seed_x: seed pixel x coordinate
        seed_y: seed pixel y coordinate
        c: connectivity (4 or 8)

    Returns:
        connected_components: set of pixels in connected component found
    """
    # Get unique color values in image
    available_colors = np.unique(image)

    # Get target color from seed pixel
    target_color = image[seed_x, seed_y]

    # Set replacement color as the other color in the binary image
    if target_color == available_colors[0]:
        replacement_color = available_colors[1]
    elif target_color == available_colors[1]:
        replacement_color = available_colors[0]

    # Initialize empty list of pixels in connected component
    connected_components = []

    # Initialize empty stack to be used in the algorithm instead of recursion
    stack = []

    # Instantiate seed pixel with its x and y coordinates and add it to the stack
    seed_pixel = Pixel(seed_x, seed_y)
    stack.append(seed_pixel)

    # Get image dimensions
    m, n = image.shape

    while len(stack) != 0:
        curr = stack.pop()

        # Check if current pixel is within image bounds
        if curr.x >= 0 and curr.x < m and curr.y >= 0 and curr.y < n:
            # Check if current pixel has target color
            if image[curr.x, curr.y] == target_color:
                # Replace current pixel color with replacement color
                image[curr.x, curr.y] = replacement_color

                # Add current pixel to connected component list if it hasn't been added yet
                if curr not in connected_components:
                    connected_components.append(curr)

                # Add neighbors of current pixel to stack according to connectivity
                if c == 4:
                    # 4-neighborhood of current pixel
                    stack.append(Pixel(curr.x - 1, curr.y))
                    stack.append(Pixel(curr.x + 1, curr.y))
                    stack.append(Pixel(curr.x, curr.y - 1))
                    stack.append(Pixel(curr.x, curr.y + 1))
                elif c == 8:
                    # 8-neighborhood of current pixel
                    stack.append(Pixel(curr.x - 1, curr.y))
                    stack.append(Pixel(curr.x + 1, curr.y))
                    stack.append(Pixel(curr.x, curr.y - 1))
                    stack.append(Pixel(curr.x, curr.y + 1))
                    stack.append(Pixel(curr.x - 1, curr.y - 1))
                    stack.append(Pixel(curr.x - 1, curr.y + 1))
                    stack.append(Pixel(curr.x + 1, curr.y - 1))
                    stack.append(Pixel(curr.x + 1, curr.y + 1))

    return connected_components


def read_input():
    """
    Reads input from stdin and loads image.

    Returns:
        image: binary image loaded from provided filename
        i_k: seed pixel x coordinate
        j_k: seed pixel y coordinate
        c: connectivity (4 or 8)
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
    # Read input parameters and load image
    image, i_k, j_k, c = read_input()

    # Find connected components using flood fill algorithm
    connected_components = flood_fill_binary(image, i_k, j_k, c)

    # Sort connected components by x coordinate and then by y coordinate
    connected_components.sort()

    # Print coordinates (i j) of pixels in connected component found
    for pixel in connected_components:
        print(f'({pixel.x} {pixel.y})', end=" ")


if __name__ == '__main__':
    main()
