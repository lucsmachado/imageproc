#!/usr/bin/env python3

"""
Title: Assignment 02 - Fourier Transform
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429
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

def read_input():
  """
  Reads input from stdin and returns the following variables:
    I: input image file path
    H: expected image file path
    i: filter index (0, 1, 2, 3, 4, 5, 6, 7 or 8)
  """
  input_image_path = str(input())
  expect_image_path = str(input())
  i = int(input())

  # Remove trailing whitespaces from image paths 
  input_image_path = input_image_path.strip()
  expect_image_path = expect_image_path.strip()

  I = imageio.imread(input_image_path)
  H = imageio.imread(expect_image_path)

  return I, H, i

def main():
  I, H, i = read_input()

  match i:
    case 0:
      # Ideal Low-pass filter
      r = int(input())
    case 1:
      # Ideal High-pass filter
      r = int(input())
    case 2:
      # Ideal Band-pass filter
      r1 = int(input())
      r2 = int(input())
    case 3:
      # Laplacian High-pass filter
      pass
    case 4:
      # Gaussian Low-pass filter
      sigma1 = int(input())
      sigma2 = int(input())
    case 5:
      # Butterworth Low-pass filter
      D0 = float(input())
      n = float(input())
    case 6:
      # Butterworth High-pass filter
      D0 = float(input())
      n = float(input())
    case 7:
      # Butterworth Band-reject filter
      pass
    case 8:
      # Butterworth Band-pass filter
      pass

  print(I.shape, H.shape, i)

if __name__ == '__main__':
  main()