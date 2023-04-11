#!/usr/bin/env python3

"""
Title: Assignment 01 - Enhancement and Superresolution
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429

11208429.py: Applies image enhancing techniques to low resolution images, 
  combines the resulting images into a higher resolution image, then compares 
  it to a reference high resolution image by outputting the Root Mean Squared Error (RMSE).

Params:
  imglow: base file name for low resolution images
  imghigh: file name (with extension) for high resolution image
  F: enhancement method identifier (0, 1, 2 or 3)
  gamma: gamma value for gamma correction (only for F = 3)

Usage: python3 11208429.py <imglow> <imghigh> <F> <gamma>
"""

import os
import numpy as np
import imageio.v2 as imageio

def root_mean_squared_error(reference_img, enhanced_img):
  """
  Computes then returns the Root Mean Squared Error (RMSE) between the reference image and the enhanced image.
  """
  M, N = reference_img.shape

  sum = 0
  for i in range(M):
    for j in range(N):
      sum += (reference_img[i][j] - enhanced_img[i][j])**2

  error = np.sqrt(sum/(M*N))

  return error

def super_resolution(imgs):
  """
  Combines the images in the list imgs into a composed image with double the resolution then returns it.
  """
  M, N = imgs[0].shape

  img_composed = np.zeros([M * 2, N * 2], dtype=int)

  # Loops through all pixels of the composed image and assigns the corresponding pixel from the original images to it.
  for i in range(M * 2):
    for j in range(N * 2):
      if i % 2 == 0:
        if j % 2 == 0:
            # When both indices are even, a pixel from the first image is assigned to it.
            img_composed[i][j] = imgs[0][int(np.floor(i/2))][int(np.floor(j/2))]
        else:
            # When the first index is even and the second is odd, a pixel from the second image is assigned to it.
            img_composed[i][j] = imgs[1][int(np.floor(i/2))][int(np.floor(j/2))]
      else:
        if j % 2 == 0:
            # When the first index is odd and the second is even, a pixel from the third image is assigned to it.
            img_composed[i][j] = imgs[2][int(np.floor(i/2))][int(np.floor(j/2))]
        else:
            # When both indices are odd, a pixel from the fourth image is assigned to it.
            img_composed[i][j] = imgs[3][int(np.floor(i/2))][int(np.floor(j/2))]
  
  return img_composed

def gamma_correction(L, gamma):
  """
  Applies gamma correction to the image L with the given gamma value and returns the result.
  """
  L_equalized = np.zeros_like(L, dtype=int)

  for index, l in enumerate(L):
    L_equalized[index] = np.floor(255 * (l / 255) ** (1/gamma))

  return L_equalized

def histogram_equalization(img, hc, num_intensity_levels=256, joint=False):
  """
  Computes the equalized image based on the cumulative histogram hc.
  """
  M, N = img.shape

  img_equalized = np.zeros([M, N], dtype=int)

  transform_function = np.zeros(num_intensity_levels, dtype=int)
  
  if (joint):
    # Computes the transform function based on the cumulative histogram of 4 different views of the same image.
    transform_function =  [hc[level] * (num_intensity_levels - 1) / (M * N * 4) for level in range(num_intensity_levels)]
  else:
    # Computes the transform function based on the cumulative histogram of a single image.
    transform_function =  [hc[level] * (num_intensity_levels - 1) / (M * N) for level in range(num_intensity_levels)]

  # Applies the transform function to the image.
  for level in range(num_intensity_levels):
    img_equalized[img == level] = transform_function[level]
  
  return img_equalized

def cumulative_histogram(L, num_intensity_levels=256, joint=False):
  """
  Computes the cumulative histogram of an image or a list of images.
  """
  hc = np.zeros(num_intensity_levels, dtype=int)

  if (joint):
    # For each intensity level, sums the number of pixels lesser than or equal to it for all images of the list L.
    for l in L:
      hc += [np.sum(l <= i) for i in range(num_intensity_levels)]
  else:
    # For each intensity level, sums the number of pixels lesser than or equal to it for the image L.
    hc = [np.sum(L <= i) for i in range(num_intensity_levels)]

  return hc
  
def joint_cumulative_histogram_equalization(L, num_intensity_levels=256):
  """
  Computes the cumulative histogram of all images in the list L, then applies histogram equalization based on it.

  Returns a list of equalized images.
  """
  L_equalized = np.zeros_like(L, dtype=int)

  hc = cumulative_histogram(L, num_intensity_levels, joint=True)

  for index, l in enumerate(L):
    L_equalized[index] = histogram_equalization(l, hc, num_intensity_levels, joint=True)
  
  return L_equalized

def single_image_cumulative_histogram_equalization(L, num_intensity_levels=256):
  """
  Computes the cumulative histogram of each image in the list L, then applies histogram equalization based on it.

  Returns a list of equalized images.
  """
  L_equalized = np.zeros_like(L, dtype=int)

  for index, l in enumerate(L):
    hc = cumulative_histogram(l, num_intensity_levels)
    L_equalized[index] = histogram_equalization(l, hc, num_intensity_levels)

  return L_equalized

def read_input():
  """
  Reads input from stdin and returns the following variables:
    L: list of low resolution images
    H: high resolution image
    F: enhancement method identifier (0, 1, 2 or 3)
    gamma: gamma value for gamma correction
  """
  imglow = str(input())
  imghigh = str(input())
  F = int(input())
  gamma = float(input())

  """
  Starting with an empty list, loops through the files in the working directory
  and appends to the list those that start with "imglow" and end with ".png"
  """
  L = []
  for file in os.listdir("./"):
    if file.startswith(imglow) and file.endswith(".png"):
      L += [imageio.imread(file)]
  
  H = imageio.imread(imghigh)

  return L, H, F, gamma

def main():
  """
  Reads expected input, calls the appropriate enhancement functions
  and prints the Root Mean Squared Error (RMSE) between the enhanced 
  image and the high resolution reference image.
  """
  L, H, F, gamma = read_input()

  match F:
    case 0:
      # No enhancement technique applied. Skips to superrresolution.
      H_composed = super_resolution(L)
    case 1:
      """
      Applies histogram equalization based on the cumulative histogram of each image,
      then performs super resolution to the enhanced images.
      """
      L_equalized = single_image_cumulative_histogram_equalization(L)
      H_composed = super_resolution(L_equalized)
    case 2:
      """
      Applies histogram equalization based on the cumulative histogram of all images,
      then performs super resolution to the enhanced images.
      """
      L_equalized = joint_cumulative_histogram_equalization(L)
      H_composed = super_resolution(L_equalized)
    case 3:
      # Applies Gamma Correction to each image, then performs super resolution to the enhanced images.
      L_equalized = gamma_correction(L, gamma)
      H_composed = super_resolution(L_equalized)

  # Computes the RMSE between the composed image and the high resolution reference image, then prints it with 4 decimal places.
  error = root_mean_squared_error(H_composed, H)
  print(f'{error:.4f}')

if __name__ == '__main__':
  main()