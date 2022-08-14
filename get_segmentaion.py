import os
import cv2
import torch
from torchvision import transforms

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results
import requests
import json

def _load_image(image_path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image



model_url = 'http://127.0.0.1:8080/predictions/skin_model'
# read and preprocess the image

image_path = '/home/tooba/Documents/codes/skin_detection/sample_image/2.jpg'
fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
image = fn_image_transform(image_path)
with torch.no_grad():
    image = image.unsqueeze(0)

# request the segmntaion api

results  = requests.post(model_url, files={'data': open(image_path, 'rb')}).text
results = json.loads(results)
results = torch.as_tensor(results)

for category, category_image, mask_image in draw_results(image[0], results, categories=['skin']):
            
            output_name = 'result.jpg'
            print(category)
            cv2.imwrite(str(output_name), category_image)
            cv2.imwrite('mask.jpg', mask_image)


