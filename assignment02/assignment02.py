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


def main():
  pass

if __name__ == '__main__':
  main()