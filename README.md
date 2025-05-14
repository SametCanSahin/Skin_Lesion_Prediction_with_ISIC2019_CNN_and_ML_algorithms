# Skin Lesion Classification Project (ISIC 2019 Dataset)

This project focuses on the classification of dermoscopic skin lesion images using a custom-trained Convolutional Neural Network (CNN). The pipeline includes dataset balancing through offline augmentation, GPU-accelerated model training, and prediction of lesion classes from single input images.

---

## 📁 Dataset & Offline Augmentation

We used the **ISIC 2019** dataset, which contains dermoscopic images labeled with 8 skin lesion types:

- `AK`, `BCC`, `BKL`, `DF`, `MEL`, `NV`, `SCC`, `VASC`

Due to class imbalance, we applied **offline data augmentation** with the following strategy:

- `NV` and `MEL` classes were randomly downsampled to **2000** images each.
- All other classes were **augmented** using custom transforms to reach **2000 images per class**.
- Augmentations used:
  - Random Zoom (`RandomScale`)
  - Contrast adjustment (`RandomBrightnessContrast`)
  - Very light Gaussian noise (`GaussNoise`)
  - Edge enhancement (`Sharpen`)

This resulted in a **balanced dataset of 16,000 images**, improving generalization and reducing bias toward dominant classes.

---

## 🧠 Model Training

We fine-tuned a **ResNet-18** model from `torchvision.models` with the following setup:

- **Input size:** 224x224 (standardized)
- **Augmentations:** Resize, Normalize (ImageNet stats)
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (`lr=1e-4`)
- **Epochs:** 10
- **Hardware:** Trained using **GPU (RTX 3050 Ti)** with CUDA acceleration
- **Class weights:** Automatically computed based on the training set to combat imbalance

Validation and test evaluations include:
- Accuracy, F1-Score
- Confusion matrix
- ROC curves for each class

---

## 🩺 Inference – Disease Prediction

A trained model can be used to predict the class of a new lesion image.

### 🔍 Example:

```python
from PIL import Image
from torchvision import transforms
import torch

# Load model
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
image = Image.open("test_lesion.jpg").convert('RGB')
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()
