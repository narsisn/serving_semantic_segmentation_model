
import cv2
import torch
from torchvision import transforms

from semantic_segmentation import models
from semantic_segmentation import load_model


def _load_image(image_path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/model_segmentation_skin_30.pth'
    model_type = 'FCNResNet101'
    image_path = 'sample_image/2.jpg'

    # image transform 
    fn_image_transform = transforms.Compose(
        [
            transforms.Lambda(lambda image_path: _load_image(image_path)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # loading model 
    model = torch.load(checkpoint_path, map_location=device)
    model = load_model(models[model_type], model)
    model.to(device).eval()

    # reading image to create input sample 
    image = fn_image_transform(image_path)
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)

    # saving the jit scripts 
    traced_model = torch.jit.trace(model,image, strict=False)
    torch.jit.save(traced_model, "jit_models/skin_model.pt")
    print(model)
