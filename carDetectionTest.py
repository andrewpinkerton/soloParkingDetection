import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Load the pre-trained Mask R-CNN model with the recommended weights
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()  # Set model to evaluation mode

# Define a transform to convert the image into tensor format
transform = T.Compose([T.ToTensor()])

# Load your image (replace 'car_image.jpg' with the path to your image)
img_path = 'images/AAlotNorth2.png'  # Make sure the path is correct
img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB mode

# Apply the transformation to the image
img_tensor = transform(img)

# Add batch dimension (model expects [batch_size, channels, height, width])
img_tensor = img_tensor.unsqueeze(0)

# Perform inference
with torch.no_grad():
    predictions = model(img_tensor)

# Extract the masks, labels, and scores
masks = predictions[0]['masks']  # Segmentation masks (one per detected object)
labels = predictions[0]['labels']  # Predicted labels
scores = predictions[0]['scores']  # Confidence scores

# COCO dataset label for car is 3, filter out other labels and low confidence scores
car_masks = [masks[i] for i in range(len(labels)) if labels[i] == 3 and scores[i] > 0.155]

# Convert the image to a NumPy array to manipulate pixel values
img_np = np.array(img)

# Define the yellow color as RGB
yellow = np.array([255, 255, 0], dtype=np.uint8)

# Apply the car masks to highlight only the car pixels in yellow
for mask in car_masks:
    # Mask is in shape [1, height, width], so we squeeze to get [height, width]
    mask = mask.squeeze().numpy()

    # Create a boolean mask where mask > 0.5 (detected car pixels)
    car_pixels = mask > 0.6

    # Set car pixels to yellow in the image
    img_np[car_pixels] = yellow

# Convert the NumPy array back to an image
img_yellow_cars = Image.fromarray(img_np)

# Display the resulting image with cars highlighted in yellow
img_yellow_cars.show()
