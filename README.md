# DeepLearning
Practice deep learning concepts
# ResNet50 Food101 Fine-Tuned Model

This repository contains a fine-tuned ResNet50 model trained on the Food-101 dataset. The model is stored as an artifact in Weights & Biases (W&B) and can be loaded directly for inference or further fine-tuning.

## Prerequisites

Ensure you have the following installed before running the code:

- Python 3.x
- PyTorch
- torchvision
- wandb
- numpy
- matplotlib

Install dependencies using:

```bash
pip install torch torchvision wandb numpy matplotlib
```

## Setup Weights & Biases (W&B)

Login to W&B and initialize the project:

```python
import wandb
wandb.login()  # Login to W&B
```

## Loading the Model from W&B

You can retrieve the fine-tuned model using the following steps:

```python
import torch
import wandb
import torchvision.models as models
import torch.nn as nn

# Initialize W&B
wandb.init(project="your_project_name")  # Replace with your W&B project name

# Load the model artifact
artifact = wandb.use_artifact("resnet50_food101_finetuned:latest", type="model")
artifact_dir = artifact.download()

# Load the model
model = models.resnet50(pretrained=False)
num_classes = 101  # Food-101 has 101 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load saved weights
model.load_state_dict(torch.load(f"{artifact_dir}/resnet50_food101_finetuned.pth", map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")
```

## Running Inference

To make predictions on a single image:

```python
from PIL import Image
import torchvision.transforms as transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = "path/to/your/image.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted Class: {predicted_class}")
```

## Notes

- Ensure you replace `"your_project_name"` with your actual W&B project name.
- Replace `"path/to/your/image.jpg"` with the actual image path for inference.
- If using GPU, move the model and input tensor to `cuda` using `.to("cuda")`.

## License

This project is for educational and research purposes.

