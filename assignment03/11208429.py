
#!/usr/bin/env python3

"""
Title: Assignment 03 - Image Descriptors
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429
"""

import sys
import numpy as np
import imageio.v2 as imageio
from scipy import ndimage


def distance(a, b):
    """
    Computes and returns the Euclidean distance between two arrays.
    """
    return np.sqrt(((a - b)**2).sum())


def knn(X0_distances, X1_distances, k=3):
    """
    Computes and returns the k nearest neighbors of a test image.

    Params:
        X0_distances: list of distances between images with class 0 and the test image
        X1_distances: list of distances between images with class 1 and the test image
        k: number of nearest neighbors to return
    """

    # Concatenate the distances into a single array, which will be used as the training sample for the classifier
    Xtrain = np.append(X0_distances, X1_distances)

    # Sort the training sample from lowest to highest
    ascending_distances = np.sort(Xtrain)

    # Get the k closest samples
    k_closest_samples = ascending_distances[:k]

    return k_closest_samples


def classify(X0_descriptors, X1_descriptors, Xtest_descriptor):
    """
    Classifies a test image in either class 0 or 1.

    Params:
        X0_descriptors: list of descriptors of images with class 0
        X1_descriptors: list of descriptors of images with class 1
        Xtest_descriptor: descriptor of the test image
    """
    # Compute the distances between the test image and all images of class 0
    X0_distances = [distance(img, Xtest_descriptor) for img in X0_descriptors]

    # Compute the distances between the test image and all images of class 1
    X1_distances = [distance(img, Xtest_descriptor) for img in X1_descriptors]

    # Get the k closest samples to the test image
    k_closest_samples = knn(
        X0_distances, X1_distances, k=3)

    # Initialize vote counters
    X0_votes = 0
    X1_votes = 0

    # For each of the k closest samples, check if it belongs to class 0 or 1 and increment the respective vote counter
    for sample in k_closest_samples:
        if sample in X0_distances:
            X0_votes += 1
        else:
            X1_votes += 1

    # Return the class with the most votes
    if (X0_votes >= X1_votes):
        return 0
    else:
        return 1


def run_tests(X0_descriptors, X1_descriptors, Xtest_descriptors):
    """
    Runs the prediction algorithm on the test set and returns the result.
    """
    # Create empty array to store predictions for all test images
    predictions = np.empty(len(Xtest_descriptors), dtype=np.uint8)

    # For each test image, predict whether it has a human or not
    for index, Xtest_descriptor in enumerate(Xtest_descriptors):
        # Classify the test image as either having (1) or not having (0) a human
        predictions[index] = classify(
            X0_descriptors, X1_descriptors, Xtest_descriptor)

    return predictions


def descriptor(m, phi):
    """
    Given the magnitude and direction of the gradient of an image, computes and returns the descriptor of the image.
    """
    # Create zero-initialized array with 9 bins to store result
    d = np.zeros(9)

    # Shift the range of angles from [-pi/2, pi/2] to [0, pi]
    phi_shift = phi + np.pi/2

    # Convert angles from radians to degrees
    phi_converted = np.degrees(phi_shift)

    # Slice the angles into 9 bins, each representing a 20 degree interval in the range [0, 180)
    phi_bins = phi_converted // 20

    # For each bin, compute the sum of the magnitudes of the gradients whose angles fall within the bin
    for i in range(9):
        # Get list of indices of pixels whose angles fall within the bin
        indices = np.argwhere(phi_bins == i)

        # For each pixel, add the magnitude of the gradient to the bin
        for index in indices:
            # Destructure pixel index
            x, y = index

            # Add the magnitude of the gradient to the bin
            d[i] += m[x, y]

    # Handle edge case where angle is 180 degrees
    edge_case_indices = np.argwhere(phi_bins == 9)

    # For each pixel whose angle is 180 degrees, add the magnitude of the gradient to the first and last bins
    for index in edge_case_indices:
        # Destructure pixel index
        x, y = index

        # Add the magnitude of the gradient to the first and last bins
        d[0] += m[x, y]
        d[8] += m[x, y]

    return d


def gradient_direction(gx, gy):
    """
    Given the gradient of an image in the x and y directions, computes and returns the direction of the gradient.
    """
    # Ignore division by zero and invalid value warnings
    np.seterr(divide='ignore', invalid='ignore')

    return np.arctan(gy / gx)


def gradient_magnitude(gx, gy):
    """
    Given the gradient of an image in the x and y directions, computes and returns the magnitude of the gradient.
    """
    # Compute the Square Root of the Sum of Squares (SRSS) of the gradients
    srss = np.sqrt(gx**2 + gy**2)

    # Normalize the SRSS
    m = srss / srss.sum()

    return m


def gradients(img):
    """
    Computes and returns the gradient of an image in the x and y directions.
    """
    # Define 3x3 Sobel kernels in x and y directions
    sobel_operator_x = [[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]]

    sobel_operator_y = [[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]]

    # Perform convolution of image with Sobel operators to obtain gradients approximations
    gx = ndimage.convolve(img, sobel_operator_x)
    gy = ndimage.convolve(img, sobel_operator_y)

    return gx, gy


def hog(img):
    """
    Computes the Histogram of Oriented Gradients (HOG) for an image and returns its descriptor.
    """
    # Compute the gradients of the image in the x and y directions
    gx, gy = gradients(img)

    # Compute the magnitude and direction of the gradient
    m = gradient_magnitude(gx, gy)
    phi = gradient_direction(gx, gy)

    # Compute the descriptor of the image
    d = descriptor(m, phi)

    return d


def image_descriptors(imgs):
    """
    Computes the image descriptors for a list of images and returns the result.
    """
    # Create empty array to store result
    result = np.empty((len(imgs), 9))

    # Compute the HOG descriptor for each image
    for index, img in enumerate(imgs):
        result[index] = hog(img)

    return result


def preprocessing(imgs):
    """
    Preprocesses a list of images and returns the result.
    """
    # Get dimensions assuming all images are of the same resolution
    M, N = imgs[0].shape[:2]

    # Create empty array to store result
    result = np.empty((len(imgs), M, N))

    # Convert each image to grayscale
    for index, img in enumerate(imgs):
        result[index] = rgbToGrayscale(img)

    return result


def rgbToGrayscale(img):
    """
    Converts an RGB image to grayscale and returns the result.
    """
    # Get image dimensions
    M, N = img.shape[:2]

    # Create empty array to store result
    result = np.empty((M, N))

    # Convert each pixel to grayscale
    for i in range(M):
        for j in range(N):
            # Get RGB values (ignoring alpha channel)
            red, green, blue = img[i, j][:3]
            # Compute the result using the luminance method
            result[i, j] = np.floor(0.299 * red + 0.587 * green + 0.114 * blue)

    return result


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

    # Read images without humans
    X0 = [imageio.imread(file) for file in lines[0]]

    # Read images with humans
    X1 = [imageio.imread(file) for file in lines[1]]

    # Read images to be classified
    Xtest = [imageio.imread(file) for file in lines[2]]

    return X0, X1, Xtest


def main():
    """
    Program entry point.
    """
    # Read input images
    X0, X1, Xtest = read_input()

    # Convert images to grayscale
    X0_grayscale = preprocessing(X0)
    X1_grayscale = preprocessing(X1)
    Xtest_grayscale = preprocessing(Xtest)

    # Compute image descriptors
    X0_descriptors = image_descriptors(X0_grayscale)
    X1_descriptors = image_descriptors(X1_grayscale)
    Xtest_descriptors = image_descriptors(Xtest_grayscale)

    predictions = run_tests(X0_descriptors, X1_descriptors, Xtest_descriptors)
    print(*predictions, sep=' ')


if __name__ == '__main__':
    main()
