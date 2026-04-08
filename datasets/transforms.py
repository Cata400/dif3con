from PIL import Image
import torchvision.transforms.functional as F

class TransformPILtoRGBTensor:
    def __call__(self, img):
        assert type(img) is Image.Image, "Input is not a PIL.Image"
        return F.pil_to_tensor(img)