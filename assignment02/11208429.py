#!/usr/bin/env python3

"""
Title: Assignment 02 - Fourier Transform
Course code: SCC0251
Year/Semester: 2023/1

Student name: Lucas Carvalho Machado
USP number: 11208429
"""

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

def normalize(values, range=1):
  """
  Normalizes the given values to the inclusive range [0, range].
  """
  min = np.min(values)
  max = np.max(values)
  return (values - min) / (max - min) * range

def apply_filter(fft, filter):
  """
  Applies the given filter to the given FFT, converts it back to the spatial domain and returns the result.
  """
  fft_filtered = np.multiply(fft, filter)

  # Shift the zero-frequency component back to the top-left corner.
  fft_filtered_unshift = np.fft.ifftshift(fft_filtered)

  # Convert the filtered FFT back to the spatial domain, then take only the real part.
  img_filtered = np.fft.ifft2(fft_filtered_unshift).real

  # Normalize the filtered image to the range [0, 255].
  img_normalized = normalize(img_filtered, 255)

  return img_normalized

def distance(p1, p2):
  """
  Computes then returns the Euclidean distance between two given points in 1 or 2 dimensions.
  """
  if isinstance(p1, (int, float, complex)) and isinstance(p2, (int, float, complex)):
    distance = np.abs(p1 - p2)
  elif isinstance(p1, tuple) and isinstance(p2, tuple):
    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
  else:
    raise TypeError("The given points must be either integers, floats, complex numbers or tuples.")
  return distance

def high_pass(shape, u, v, r):
  """
  Returns the value of the ideal high-pass filter at the given coordinates.
  """
  P, Q = shape

  # Compute the distance between the current coordinates and the center of the image.
  D = distance((u, v), (P/2, Q/2))

  if D <= r:
    # If the distance is less than or equal to the radius, the filter value is 0 (the frequency is blocked).
    return 0
  else:
    # Otherwise, the filter value is 1 (the frequency is allowed).
    return 1

def low_pass(shape, u, v, r):
  """
  Returns the value of the ideal low-pass filter at the given coordinates.
  """
  P, Q = shape

  # Compute the distance between the current coordinates and the center of the image.
  D = distance((u, v), (P/2, Q/2))

  if D <= r:
    # If the distance is less than or equal to the radius, the filter value is 1 (the frequency is allowed).
    return 1
  else:
    # Otherwise, the filter value is 0 (the frequency is blocked).
    return 0

def band_pass(shape, u, v, r1, r2):
  """
  Returns the value of the ideal band-pass filter at the given coordinates.
  """
  return low_pass(shape, u, v, r1) - low_pass(shape, u, v, r2)

def laplacian_high_pass(shape, u, v):
  """
  Returns the value of the Laplacian high-pass filter at the given coordinates.
  """
  P, Q = shape

  # Compute the squared distance between the current coordinates and the center of the image.
  squared_distance_center = distance((u, v), (P/2, Q/2))**2

  # Compute and return the filter value.
  return 4 * np.pi**2 * squared_distance_center

def gaussian_low_pass(shape, u, v, sigma1, sigma2):
  """
  Returns the value of the Gaussian low-pass filter at the given coordinates.
  """
  P, Q = shape

  # Compute the squared distance for each direction between the coordinate and the center of the image.
  squared_distance_center_row = distance(u, P/2)**2
  squared_distance_center_col = distance(v, Q/2)**2

  # Compute twice the squared standard deviations for each direction.
  twice_squared_std_dev_row = 2 * sigma1**2
  twice_squared_std_dev_col = 2 * sigma2**2

  # Compute the x value by combining the results for each direction.
  row = squared_distance_center_row / twice_squared_std_dev_row
  col = squared_distance_center_col / twice_squared_std_dev_col
  x = row + col

  # Compute and return the filter value.
  return np.exp(-1*x)

def butterworth_low_pass(shape, u, v, D0, n):
  """
  Returns the value of the Butterworth low-pass filter at the given coordinates.
  """
  P, Q = shape

  # Compute the distance between the current coordinates and the center of the image.
  D = distance((u, v), (P/2, Q/2))

  # Compute and return the filter value.
  return 1 / (1 + (D/D0)**(2*n))

def butterworth_high_pass(shape, u, v, D0, n):
  """
  Returns the value of the Butterworth high-pass filter at the given coordinates.
  """
  return 1 - butterworth_low_pass(shape, u, v, D0, n)

def get_filter(shape, filter_fn, *args):
  """
  Returns a filter of the given shape using the given filter function.
  """
  P, Q = shape
  filter = np.zeros((P, Q), dtype=np.float32)

  # Loop over each pixel in the filter and compute its value.
  for u in range(P):
    for v in range(Q):
      filter[u, v] = filter_fn(shape, u, v, *args)

  return filter

def get_fft(img):
  """
  Computes then returns the Fast Fourier Transform of the given image.
  """
  fft = np.fft.fft2(img)

  # Shift the zero-frequency component to the center of the spectrum.
  fft_shift = np.fft.fftshift(fft)

  return fft_shift

def filter_image(img, filter_fn, *args):
  """
  Applies the given filter function to the given image and returns the result.
  """
  # Compute the FFT of the image.
  fft_shift = get_fft(img)

  # Get the shape of the image.
  P, Q = fft_shift.shape

  # Get the filter.
  filter = get_filter((P, Q), filter_fn, *args)

  # Apply the filter to the image.
  img_filtered = apply_filter(fft_shift, filter)

  return img_filtered

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
      # Read the radius from stdin.
      r = int(input())
      H_hat = filter_image(I, low_pass, r)
    case 1:
      # Ideal High-pass filter
      # Read the radius from stdin.
      r = int(input())
      H_hat = filter_image(I, high_pass, r)
    case 2:
      # Ideal Band-pass filter
      # Read the inner and outer radius from stdin.
      r1 = int(input())
      r2 = int(input())
      H_hat = filter_image(I, band_pass, r1, r2)
    case 3:
      # Laplacian High-pass filter
      H_hat = filter_image(I, laplacian_high_pass)
    case 4:
      # Gaussian Low-pass filter
      # Read the standard deviations from stdin.
      sigma1 = int(input())
      sigma2 = int(input())
      H_hat = filter_image(I, gaussian_low_pass, sigma1, sigma2)
    case 5:
      # Butterworth Low-pass filter
      # Read the cutoff frequency and order from stdin.
      D0 = float(input())
      n = float(input())
      H_hat = filter_image(I, butterworth_low_pass, D0, n)
    case 6:
      # Butterworth High-pass filter
      # Read the cutoff frequency and order from stdin.
      D0 = float(input())
      n = float(input())
      H_hat = filter_image(I, butterworth_high_pass, D0, n)
    case 7:
      # Butterworth Band-reject filter
      pass
    case 8:
      # Butterworth Band-pass filter
      pass

  # Compute the Root Mean Squared Error (RMSE) between the expected image and the enhanced image, then print it with 4 decimal places.
  error = root_mean_squared_error(H_hat, H)
  print(f'{error:.4f}')

if __name__ == '__main__':
  main()