import torch
from torchvision import transforms
from PIL import Image
from muse import MaskGitVQGAN

# Load the pre-trained vq model from the hub
vq_model = MaskGitVQGAN.from_pretrained("openMUSE/maskgit-vqgan-imagenet-f16-256")

# encode and decode images using
encode_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)
image = Image.open("/data/lychen/jzxu/muse_cleaned_version/data/test_image.png") #
pixel_values = encode_transform(image).unsqueeze(0)

print(100*"-")
print(pixel_values)

image_tokens = vq_model.encode(pixel_values)
print(100*"-")
print(image_tokens)

# rec_image = vq_model.decode(image_tokens)

# # Convert to PIL images
# rec_image = 2.0 * rec_image - 1.0
# rec_image = torch.clamp(rec_image, -1.0, 1.0)
# rec_image = (rec_image + 1.0) / 2.0
# rec_image *= 255.0
# rec_image = rec_image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
# pil_images = [Image.fromarray(image) for image in rec_image]