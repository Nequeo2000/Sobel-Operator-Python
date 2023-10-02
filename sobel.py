import numpy as np
import math

def sobel(image: np.ndarray):
    sobelImage = np.zeros(shape=image.shape)
    for i in range(image.shape[0]):
        for y in range(image.shape[1]):
            v1 = 2*image[max((i-1),0)][y]+image[max((i-1),0)][max((y-1),0)]+image[max((i-1),0)][min((y+1),image.shape[1]-1)]
            v2 = 2*image[min((i+1),image.shape[0]-1)][y]+image[min((i+1),image.shape[0]-1)][max((y-1),0)]+image[min((i+1),image.shape[0]-1)][min((y+1),image.shape[1]-1)]
            v3 = 2*image[i][max((y-1),0)]+image[max((i-1),0)][max((y-1),0)]+image[min((i+1),image.shape[0]-1)][max((y-1),0)]
            v4 = 2*image[i][min((y+1),image.shape[1]-1)]+image[max((i-1),0)][min((y+1),image.shape[1]-1)]+image[min((i+1),image.shape[0]-1)][min((y+1),image.shape[1]-1)]
            
            sobelImage[i][y] = math.sqrt(math.pow(v2-v1,2)+math.pow(v4-v3,2))
    return sobelImage

def fastSobel(image: np.ndarray):
    if np.max(image)-np.min(image) > 10:
        if image.shape[0] >= 4 or image.shape[1] >= 4:
            imageTL = image[0:int(image.shape[0]/2), 0:int(image.shape[1]/2)]
            imageTR = image[int(image.shape[0]/2):image.shape[0], 0:int(image.shape[1]/2)]
            imageBR = image[int(image.shape[0]/2):image.shape[0], int(image.shape[1]/2):image.shape[1]]
            imageBL = image[0:int(image.shape[0]/2), int(image.shape[1]/2):image.shape[1]]

            image[0:int(image.shape[0]/2), 0:int(image.shape[1]/2)] = fastSobel(image= imageTL)
            image[int(image.shape[0]/2):image.shape[0], 0:int(image.shape[1]/2)] = fastSobel(image= imageTR)
            image[int(image.shape[0]/2):image.shape[0], int(image.shape[1]/2):image.shape[1]] = fastSobel(image= imageBR)
            image[0:int(image.shape[0]/2), int(image.shape[1]/2):image.shape[1]] = fastSobel(image= imageBL)
            
            return image
        else:
            sobelImage = np.zeros(shape=image.shape)
            for i in range(image.shape[0]):
                for y in range(image.shape[1]):
                    v1 = 2*image[max((i-1),0)][y]
                    v2 = 2*image[min((i+1),image.shape[0]-1)][y]
                    v3 = 2*image[i][max((y-1),0)]
                    v4 = 2*image[i][min((y+1),image.shape[1]-1)]
                    
                    sobelImage[i][y] = math.sqrt(math.pow(v2-v1,2)+math.pow(v4-v3,2))
            return sobelImage
    else:
        return np.zeros(shape=image.shape)