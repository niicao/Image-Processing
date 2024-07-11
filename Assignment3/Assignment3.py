### GENERAL INFORMATION #######################################################
#   Name:           Nicholas Yudi Kurita Ikai                                 #
#   USP number:     13671852                                                  #
#   course code:    SCC0251                                                   #
#   Year/Semestre:  2024/1                                                    #
#   Title:          Assignment 3 : Morphology + Color                         #
###############################################################################

import math
import numpy as np
import os
import imageio.v3 as imageio

def find_files(basename: str):
    files = []
    for filename in os.listdir(os.getcwd()):
        if os.path.basename(filename).startswith(basename):
            files.append(os.path.join(os.getcwd(), filename))
    return files

def read_image(image_path: str):
    image = imageio.imread(image_path)
    return image

def thresholding(img,  L):
  #cria uma matriz de zeros para receber a imagem binaria
  img_thold = np.ones(img.shape).astype(np.uint8)
  #realiza a binarização da matriz a partir do valor L
  img_thold[np.where(img <= L)] = 0
  img_thold[np.where(img > L)] = 1
  return img_thold

def otsu(img: np.ndarray, c) -> np.ndarray:
  num_pixels = img.shape[0] * img.shape[1]
  #cria uma lista para armazenar os valores de variancia
  max_var = []
  #calcula o histograma da imagem que passou pelo threshold (imagem binaria)
  hist_thold = np.histogram(img, bins = 256, range=(0,256))[0]
  #calcula o valor da variancia para cada threshold
  for value in range(1, c):
    img_ti = thresholding(img, value)
    #calcula o peso de background e foreground
    weight_bg = np.sum(hist_thold[:value]) / float(num_pixels)
    weight_fg = np.sum(hist_thold[value:]) / float(num_pixels)  

    #calcula as medias de background e foreground
    mean_bg = np.sum(np.arange(value) * hist_thold[:value]) / (np.sum(hist_thold[:value]) + 1e-10)  # soma 1e-10 para evitar divisão por zero
    mean_fg = np.sum(np.arange(value, c) * hist_thold[value:]) / (np.sum(hist_thold[value:]) + 1e-10)  #  soma 1e-10 para evitar divisão por zero
   
    #calcula a variancia entre classes e adiciona o valor na lista
    sigma_between = weight_bg*weight_fg* ((mean_bg - mean_fg) ** 2)
    max_var.append(sigma_between)
    
  #computa a imagem binarizada a partir do valor otimo calculado pelo Otsu
  threshold_img = thresholding(img, np.argmax(max_var))
  return threshold_img




def erosion(img, M, N):
    img_er = img.copy()

    for i in range(1, M-1):
      for j in range(1, N-1):
        # se o valor minimo for 0, entao significa que há pelo menos um 0 no kernel, entao o pixel central é substituido por 0
        img_er[i][j] = np.min(img[i-1:i+2,j-1:j+2])

    return img_er

def dilation(img, M, N):
    img_di = img.copy()

    for i in range(1, M-1):
      for j in range(1, N-1):
        # se o valor maximo for 1, significa que há pelo menos um 1 no kernel, entao o pixel central é substituido por 1
        img_di[i][j] = np.max(img[i-1:i+2,j-1:j+2])

    return img_di

def filter_gaussian(P, Q):
    s1 = P
    s2 = Q

    D = np.zeros([P, Q])  # Compute Distances
    for u in range(P):
        for v in range(Q):
            x = (u-(P/2))**2/(2*s1**2) + (v-(Q/2))**2/(2*s2**2)
            D[u, v] = np.exp(-x)
    return D

def map_value_to_color(value, min_val, max_val, colormap):
    # Scale the value to the range [0, len(colormap) - 1]
    scaled_value = (value - min_val) / (max_val - min_val) * (len(colormap) - 1)
    # Determine the two closest colors in the colormap
    idx1 = int(scaled_value)
    idx2 = min(idx1 + 1, len(colormap) - 1)
    # Interpolate between the two colors based on the fractional part
    frac = scaled_value - idx1
    color = [
        (1 - frac) * colormap[idx1][0] + frac * colormap[idx2][0],
        (1 - frac) * colormap[idx1][1] + frac * colormap[idx2][1],
        (1 - frac) * colormap[idx1][2] + frac * colormap[idx2][2]
    ]
    return color

def rms_error(img, out):
    M,N = img.shape
    error = ((1/(M*N))*np.sum((img-out)**2))**(1/2)
    return error

def main():
    #leitura do input
    image1 = read_image(str(input().strip()))
    image_ref = read_image(str(input().strip()))
    vector_operations = list(map(int, input().split()))
    
    #converte a imagem para grayscale caso ela seja RGB
    if len(image1.shape) > 2:
        image_gray = np.dot(image1, [0.2989, 0.5870, 0.1140]).astype(np.int64)
    else:
        image_gray = image1
        
    #recebe a imagem binarizada com o valor mais eficiente
    bin_image = otsu(image_gray, 256)
    
    M,N = bin_image.shape

    # realiza as operacoes pedidas
    for op in vector_operations:
        if op == 1:
            bin_image = erosion(bin_image, M, N)
        elif op == 2:
            bin_image = dilation(bin_image, M, N)
    

        #Espectro Visível
    heatmap_colors = [
        [1, 0, 1],   # Pink
        [0, 0, 1],   # Blue
        [0, 1, 0],   # Green
        [1, 1, 0],   # Yellow
        [1, 0, 0]    # Red
    ]
    alpha = 0.30
    
    mask = bin_image
    color_distribution = filter_gaussian(M,N)
    min_val = np.min(np.array(color_distribution))
    max_val = np.max(np.array(color_distribution))
    
    heatmap_image = np.zeros((M, N, 3))
    for i in range(M):
      for j in range(N):
          heatmap_image[i, j] = map_value_to_color(color_distribution[i, j], min_val, max_val, heatmap_colors)
    
    img_color = np.ones([M, N, 3]) #Imagem RGB vazia
    indexes = np.where(mask==0)
    img_color[indexes] = heatmap_image[indexes]
    
    #passa a imagem colorida para o invervalo (0,255)
    img_color = ((img_color*255) / np.max(img_color)).astype(int)
    
    #normaliza a imagem cinza
    image_gray_norm = (image_gray / np.max(image_gray) * 255).astype(np.uint8)
    
    mixed_image = np.zeros([M,N,3])
    
    #cria uma imagem de 3 canais que recebe a imagem cinza em todos eles para que ela possa ser misturada com a imagem colorida
    gray_img_3dim = np.ones([M,N,3])
    for i in range (0,3):
        gray_img_3dim[:,:,i] = image_gray_norm[:, :]
        
    #gera a imagem final
    mixed_image = ((1-alpha)*gray_img_3dim + alpha*img_color)
    
    
  #Calcula o erro para cada canal de cor
    error_R = rms_error(mixed_image[:,:,0], image_ref[:,:,0])
    error_G = rms_error(mixed_image[:,:,1], image_ref[:,:,1])
    error_B = rms_error(mixed_image[:,:,2], image_ref[:,:,2])
    error = (error_R + error_G + error_B)/3
    print(f"{error:.4f}")
    
if __name__ == '__main__':
    main()