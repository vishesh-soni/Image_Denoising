import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf
# from tensorflow.keras

from keras.models import load_model


def add_gaussian_noise(image, mean=0, stddev=25):
    noise = np.random.normal(mean, stddev, image.shape).astype('uint8')
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image

def apply_gaussian_filter(image, sigma=1):
    filtered_image = gaussian_filter(image, sigma=sigma)
    return filtered_image

def apply_gaussian_filter_with_noise(image_path, noise_mean, noise_stddev, sigma):
    # Load the image using PIL
    image = cv2.imread(image_path)
    # print(image.shape)
#     image = image.reshape(-1,3044,/ 4048, 3)
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = image.reshape(-1,64,64,3)
    
    # Add Gaussian noise
    print(image.shape)
    noisy_image = add_gaussian_noise(image, noise_mean, noise_stddev)
    print('step 1')

    # Apply Gaussian filter
    filtered_image = apply_gaussian_filter(noisy_image, sigma)
    print('step 2')

    return image,noisy_image, filtered_image

def cnn_data(file_path,std):
    noise_mean = 0  # Mean of Gaussian noise
    noise_stddev = std  # Standard deviation of Gaussian noise
    sigma = 1  # Sigma for the Gaussian filter
    image_path = file_path 
    image,noisy_image, filtered_image = apply_gaussian_filter_with_noise(image_path, noise_mean, noise_stddev, sigma)
    print('step 3')
    return image,noisy_image,filtered_image

def predict_img(noisy_image,model):
    pred = noisy_image
    pred = pred.reshape(-1,224,224,1)
    predicted_image = model.predict(pred)
    return predicted_image

