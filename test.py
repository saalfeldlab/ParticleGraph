import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load a pre-trained Mask R-CNN model from torchvision
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image transformation (resizing, normalization, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to tensor and normalize to [0, 1]
])


# Function to load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    img_tensor = transform(img)  # Apply the transformations
    return img, img_tensor


# Load your input image
image_path = "/groups/saalfeld/home/allierc/signaling/RatCity/images/0-1012.jpg"
img, img_tensor = load_image(image_path)

# Add batch dimension (since the model expects a batch of images)
img_tensor = img_tensor.unsqueeze(0)

# Perform inference on the image
with torch.no_grad():
    prediction = model(img_tensor)

# Extract prediction results
masks = prediction[0]['masks']  # Segmentation masks for each object detected
labels = prediction[0]['labels']  # Labels for the detected objects
scores = prediction[0]['scores']  # Confidence scores for each prediction

# Filter out low-confidence predictions (e.g., score < 0.5)
threshold = 0.5
masks = masks[scores > threshold]
labels = labels[scores > threshold]

# Convert the masks to numpy for visualization
masks = masks.mul(255).byte().cpu().numpy()

# Plot the original image and overlay the masks
plt.figure(figsize=(10, 10))
plt.imshow(img)

# Overlay each mask on the image
for i in range(masks.shape[0]):
    mask = masks[i, 0]  # Get the mask for the i-th object (single channel)
    mask_img = Image.fromarray(mask)

    # You can optionally adjust the transparency of each mask
    plt.imshow(mask_img, alpha=0.5)  # Apply mask with transparency

plt.axis('off')  # Turn off axis labels
plt.show()
