# image_upload/views.py
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from image_denoise import cnn_data, predict_img
from keras.models import load_model
# from django.utils.encoding import base64_encode
import cv2
import base64
import os
from bilateral import *
from PIL import Image

def image_upload(request):
    form = ImageUploadForm()
    return render(request, 'image_upload.html', {'form': form})

def image_list(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save(commit=False)
            uploaded_image.save()

            # noisy_image_path = os.path.join('media', 'noisy_images', uploaded_image.image.name)
            # generate_noisy_image(image_path, noisy_image_path)

            model_path = "Model\gfgModel.h5"
            model = load_model(model_path, compile=False)
            image_filename = str(request.FILES['image'])
            image_path = os.path.join('media', 'images', image_filename)
            # print(image_path)

            image,noisy_image, filtered_image = cnn_data(image_path, 10)
            pred = predict_img(noisy_image, model)
            reshaped_pred = pred.reshape(224,224)
            # kernel_size = 5
            # noise_var = 10
            reshaped_pred = bilateral_filter(reshaped_pred,15)
            # gamma = 0.5  # Adjust this value to control the contrast

            # # Apply gamma correction to decrease contrast
            # reshaped_pred = decrease_contrast(reshaped_pred, gamma)

            _, image_encoded = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(image_encoded).decode('utf-8')
            _, noisy_image_encoded = cv2.imencode('.png', noisy_image)
            noisy_image_base64 = base64.b64encode(noisy_image_encoded).decode('utf-8')
            _, reshaped_pred_encoded = cv2.imencode('.png', reshaped_pred)
            reshaped_pred_base64 = base64.b64encode(reshaped_pred_encoded).decode('utf-8')

            return render(request, 'image_list.html', {'original_image': image_base64, 'noisy_image': noisy_image_base64,'reshaped_pred': reshaped_pred_base64})

    return render(request, 'image_list.html')
