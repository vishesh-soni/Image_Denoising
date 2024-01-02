import cv2
import numpy as np
def bilateral_filter(noisy_image,std):
    # Apply bilateral filter
    filtered_image = cv2.bilateralFilter(noisy_image, d=5, sigmaColor=12, sigmaSpace=15)
    
    return filtered_image
