import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the YOLOv5 model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img_path = 'cars.jpg'
img = Image.open(img_path)

results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()