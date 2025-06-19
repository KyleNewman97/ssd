import torch

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from ssd import SSD

if __name__ == "__main__":
    image = Image.open("/mnt/data/code/ssd/data/dog.jpeg")
    image = image.resize((300, 300))
    image_tensor = pil_to_tensor(image).type(torch.float).cuda()
    batch_tensor = image_tensor[None, :, :, :]
    batch_tensor /= 255

    print("Image size:", image_tensor.shape)
    print("Batch size:", batch_tensor.shape)

    model = SSD(num_classes=2)
    print(model)

    result = model.forward(batch_tensor)
    print(result.shape)
