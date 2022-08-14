'''
Module for image segmentaion default handler
''' #
import torch
import torch.nn.functional as F
from torchvision import transforms

from ts.torch_handler.vision_handler import VisionHandler
import cv2
from PIL import Image
import io
import base64
import numpy as np


class ImageSegmenter(VisionHandler):
    """
    ImageSegmentaion handler class. This handler takes an image
    and returns the segmented parts of that image.
    """


    def _load_image(self,image):
        
        image_width = (image.shape[1] // 32) * 32
        image_height = (image.shape[0] // 32) * 32

        image = image[:image_height, :image_width]
        return image

    # image transform 
    image_processing =  transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = np.array(Image.open(io.BytesIO(image)))
                image = self._load_image(image)
                image = self.image_processing(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)

    def postprocess(self, data):
        # Returning the class for every pixel makes the response size too big
        # (> 24mb). Instead, we'll only return the top class for each image
        data = data['out']
        data = torch.sigmoid(data)
        data = data > 0.5
        
        return data.tolist()

    
 
    



