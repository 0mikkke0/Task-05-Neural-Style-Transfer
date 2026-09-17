import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading and preprocessing transformation
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Load pre-trained VGG19 model for feature extraction
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze all VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

print("Neural Style Transfer pipeline initialized successfully with VGG19!")
