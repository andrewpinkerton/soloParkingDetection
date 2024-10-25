import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load parking spot regions from regions.p
with open('regions.p', 'rb') as f:
    parking_spots = pickle.load(f)

# Load the pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define a transform to convert the image into tensor format
transform = T.Compose([T.ToTensor()])

# Load your image (replace 'car_image.jpg' with the path to your image)
img_path = 'images/AAlotNorth2.png'
img = Image.open(img_path).convert("RGB")

# Apply the transformation to the image
img_tensor = transform(img).unsqueeze(0)  # Add batch dimensio

# Perform inference
with torch.no_grad():
    predictions = model(img_tensor)

# Extract the masks, labels, and scores
masks = predictions[0]['masks']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Filter for car masks (COCO label for car is 3)
car_masks = [masks[i] for i in range(len(labels)) if labels[i] == 3 and scores[i] > 0.155]

# Convert the image to a NumPy array
img_np = np.array(img)

# Define yellow color for highlighting
yellow = np.array([255, 255, 0], dtype=np.uint8)

# Highlight car pixels in yellow
for mask in car_masks:
    mask = mask.squeeze().numpy() > 0.5  # Create a boolean mask
    img_np[mask] = yellow  # Set car pixels to yellow

# Convert back to an image
img_yellow_cars = Image.fromarray(img_np)

# Optional: Overlay parking spots
plt.figure(figsize=(12, 6))
plt.imshow(img_np)
for spot in parking_spots:
    spot_coords = np.array(spot)
    x_coords = spot_coords[:, 0]
    y_coords = spot_coords[:, 1]
    plt.fill(x_coords, y_coords, color='red', alpha=0.3)
plt.title("Parking Spots Overlay with Highlighted Cars")
plt.axis('off')
plt.show()
