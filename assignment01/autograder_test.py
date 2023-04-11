#!/usr/bin/env python3

"""
Title: Assignment 01 - Enhancement and Superresolution
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429

11208429.py: Applies image enhancing techniques to low resolution images, 
  combines the resulting images into a higher resultion image, then compares 
  it to a reference high resolution image by outputting the Root Mean Squared Error (RMSE).

Params:
  imglow: base file name for low resolution images
  imghigh: file name (with extension) for high resolution image
  F: enhancement method identifier (0, 1, 2 or 3)
  gamma: gamma value for gamma correction (only for F = 3)

Usage: python3 11208429.py <imglow> <imghigh> <F> <gamma>
"""

import sys
import os
import numpy as np
import imageio.v2 as imageio

def root_mean_squared_error(reference_img, enhanced_img):
  M, N = reference_img.shape

  sum = 0
  for i in range(M):
    for j in range(N):
      sum += (reference_img[i][j] - enhanced_img[i][j])**2

  error = np.sqrt(sum/(M*N))

  return error

def super_resolution(imgs):
  M, N = imgs[0].shape

  img_composed = np.zeros([M * 2, N * 2], dtype=int)

  for i in range(M * 2):
    for j in range(N * 2):
      if i % 2 == 0:
        if j % 2 == 0:
            # first image
            img_composed[i][j] = imgs[0][int(np.floor(i/2))][int(np.floor(j/2))]
        else:
            # second image
            img_composed[i][j] = imgs[1][int(np.floor(i/2))][int(np.floor(j/2))]
      else:
        if j % 2 == 0:
            # third image
            img_composed[i][j] = imgs[2][int(np.floor(i/2))][int(np.floor(j/2))]
        else:
            # fourth image
            img_composed[i][j] = imgs[3][int(np.floor(i/2))][int(np.floor(j/2))]
  
  return img_composed

def gamma_correction(L, gamma):
  L_equalized = np.zeros_like(L, dtype=int)

  for index, l in enumerate(L):
    L_equalized[index] = np.floor(255 * (l / 255) ** (1/gamma))

  return L_equalized

def histogram_equalization(img, hc, num_intensity_levels=256, joint=False):
  M, N = img.shape

  img_equalized = np.zeros([M, N], dtype=int)

  transform_function = np.zeros(num_intensity_levels, dtype=int)
  
  if (joint):
    transform_function =  [hc[level] * (num_intensity_levels - 1) / (M * N * 4) for level in range(num_intensity_levels)]
  else:
    transform_function =  [hc[level] * (num_intensity_levels - 1) / (M * N) for level in range(num_intensity_levels)]

  for level in range(num_intensity_levels):
    img_equalized[img == level] = transform_function[level]
  
  return img_equalized

def cumulative_histogram(L, num_intensity_levels=256, joint=False):
  hc = np.zeros(num_intensity_levels, dtype=int)

  if (joint):
    for l in L:
      hc += [np.sum(l <= i) for i in range(num_intensity_levels)]
  else:
    hc = [np.sum(L <= i) for i in range(num_intensity_levels)]

  return hc
  
def joint_cumulative_histogram_equalization(L, num_intensity_levels=256):
  L_equalized = np.zeros_like(L, dtype=int)

  hc = cumulative_histogram(L, num_intensity_levels, joint=True)

  for index, l in enumerate(L):
    L_equalized[index] = histogram_equalization(l, hc, num_intensity_levels, joint=True)
  
  return L_equalized

def single_image_cumulative_histogram_equalization(L, num_intensity_levels=256):
  L_equalized = np.zeros_like(L, dtype=int)

  for index, l in enumerate(L):
    hc = cumulative_histogram(l, num_intensity_levels)
    L_equalized[index] = histogram_equalization(l, hc, num_intensity_levels)

  return L_equalized

def read_input():
  imglow = str(input())
  imghigh = str(input())
  F = int(input())
  gamma = float(input())

  L = []
  for file in os.listdir("./"):
    if file.startswith(imglow) and file.endswith(".png"):
      L += [imageio.imread(file)]
  
  H = imageio.imread(imghigh)

  return L, H, F, gamma

def main():
  L, H, F, gamma = read_input()

  match F:
    case 0:
      H_composed = super_resolution(L)
    case 1:
      L_equalized = single_image_cumulative_histogram_equalization(L)
      H_composed = super_resolution(L_equalized)
    case 2:
      L_equalized = joint_cumulative_histogram_equalization(L)
      H_composed = super_resolution(L_equalized)
    case 3:
      L_equalized = gamma_correction(L, gamma)
      H_composed = super_resolution(L_equalized)

  error = root_mean_squared_error(H_composed, H)
  print(f'{error:.4f}')

if __name__ == '__main__':
  main()