import numpy as np

def sobel(image: np.ndarray):
    """
    Parameters
    ----------
        image: grayscale image in the form of a numpy ndarray containing value from 0 to 255 

    Returns
    -------
        ndarray feature mask of dimensions (n-2),(m-2)
    """
    featureMask = np.ndarray(shape=(image.shape[0]-2,image.shape[1]-2))

    for i in range(1,image.shape[0]-1,1):
        for y in range(1,image.shape[1]-1,1):
            pixelMatrix = image[i-1:i+2 , y-1:y+2]
            gx = np.sum(np.multiply(pixelMatrix, np.matrix([[1,0,-1],
                                                            [2,0,-2],
                                                            [1,0,-1]])))
            gy = np.sum(np.multiply(pixelMatrix, np.matrix([[1,2,1],
                                                            [0,0,0],
                                                            [-1,-2,-1]])))

            featureMask[i-1][y-1] = np.sqrt(np.power(gx,2)+np.power(gy,2))
    return featureMask