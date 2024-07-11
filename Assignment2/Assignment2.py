# Nicholas Yudi Kurita Ikai 13671852 - SCC0251 2024/1 - Assignment 2: fourier transform & filtering in frequency domain

import imageio.v3 as imageio
import numpy as np
import os
import math

def find_files(basename: str):
    files = []
    for filename in os.listdir(os.getcwd()):
        if os.path.basename(filename).startswith(basename):
            files.append(os.path.join(os.getcwd(), filename))
    return files

def read_image(image_path: str):
    image = imageio.imread(image_path, mode='L')
    return image

def rmse(old_image, highres_image):
  lines, columns = old_image.shape
  cum_sum = 0
  for i in range(lines):
    for j in range(columns):
      cum_sum += pow((old_image[i][j] - highres_image[i][j]), 2)

  cum_sum /= lines*columns
  return np.sqrt(cum_sum)

def fourier_spectrum(image:np.ndarray):
  return np.fft.fftshift(np.fft.fft2(image))

def inverse_fft(image:np.ndarray):
  return np.fft.ifft2(np.fft.ifftshift(image))

def normalize_image(image:np.ndarray):
  image1 = image - image.min()
  image1 = (image1/image1.max())*255
  return image1

def low_pass_filter(image:np.ndarray, radius:float):
  P, Q = image.shape
  H = np.zeros((P,Q), dtype = int)
  for u in range(P):
      for v in range(Q):
          D = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
          if D <= radius:
              H[u,v] = 1
          else:
              H[u,v] = 0
  return H

def high_pass_filter(image:np.ndarray, radius:float):
  P, Q = image.shape
  H = np.zeros((P,Q), dtype = int)
  for u in range(P):
      for v in range(Q):
          D = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
          if D >= radius:
              H[u,v] = 1
          else:
              H[u,v] = 0
  return H

def band_stop_filter(image:np.ndarray, radius0:float, radius1:float):
  P, Q = image.shape
  H = np.zeros((P, Q), dtype = int)
  for u in range(P):
    for v in range(Q):
      D = np.sqrt((u-P/2)**2 + (v-Q/2)**2)
      if D <= radius0 and D >= radius1:
        H[u,v] = 0
      else:
        H[u,v] = 1
  return H

def laplacian_filter(image:np.ndarray):
  P, Q = image.shape
  H = np.zeros((P,Q), dtype = int)
  for u in range(P):
    for v in range(Q):
      H[u,v] = -4*(np.pi**2)*((u - (P/2))**2 + (v - (Q/2))**2)
  return H

def gama_filter(image:np.ndarray, gama1:int, gama2:int):
  P, Q = image.shape
  H = np.zeros((P,Q))
  for u in range(P):
    for v in range(Q):
      x = ( (u - (P/2))**2 / (2*(gama1**2)) + (v - (Q/2))**2 / (2*(gama2)**2))
      H[u, v] = math.exp(-x)
  return H

def main():
  # Read the files and turn them into images (arrays) and compute their fourier spectrum
  image1 = read_image(str(input()).strip())
  image_ref = read_image(str(input()).strip())
  fourier_transf = fourier_spectrum(image1)
  
  index_input = int(input())
  
  #For every case, compute the filter and multiply it by the fourier spectrum of the image
  match index_input:
    case 0:
      radius = float(input())
      low_filter = low_pass_filter(image1, radius)
      filtered_freq = fourier_transf * low_filter
      
    case 1:
      radius = float(input())
      high_filter = high_pass_filter(image1, radius)
      filtered_freq = fourier_transf * high_filter

    case 2:
      r0 = float(input())
      r1 = float(input())
      bStop_filter = band_stop_filter(image1, r0, r1)
      filtered_freq = fourier_transf * bStop_filter

    case 3:
      filter = laplacian_filter(image1)
      filtered_freq = fourier_transf * filter

    case 4:
      gama1 = float(input())
      gama2 = float(input())
      gFilter = gama_filter(image1, gama1, gama2)
      filtered_freq = fourier_transf * gFilter


  # Compute the inverse fourier transform of the filtered image, extract the real part and normalize it
  image_filtered = np.real(inverse_fft(filtered_freq))
  image_filtered = normalize_image(image_filtered)

  print("%.4f" % rmse(image_filtered, image_ref))

if __name__ == '__main__':
  main()