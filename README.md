# 🧠 Skin Lesion Classification Project (ISIC 2019 Dataset)

This project focuses on the classification of dermoscopic skin lesion images using a custom-trained Convolutional Neural Network (CNN). The pipeline includes dataset balancing through offline augmentation, GPU-accelerated model training, and disease prediction from input images.

---

## 👥 Team Members

- **Zeki Bozdoğanlı**
- **Murat Can Mutlu**
- **Samet Can Şahin**

---

## 📁 Dataset & Offline Augmentation

We used the **ISIC 2019** dataset, which contains dermoscopic images labeled with 8 skin lesion types:

- `AK`, `BCC`, `BKL`, `DF`, `MEL`, `NV`, `SCC`, `VASC`

To address class imbalance, we performed **offline data augmentation** with the following strategy:

- `NV` and `MEL` classes were randomly downsampled to **2000 images each**.
- All other classes were augmented to reach **2000 samples per class**.
- Applied augmentations:
  - Random Zoom (`RandomScale`)
  - Contrast Adjustment (`RandomBrightnessContrast`)
  - Very light Gaussian Noise (`GaussNoise`, `std_range=(0.05, 0.1)`)
  - Edge Enhancement (`Sharpen`, `alpha=(0.3, 0.6)`)

This resulted in a **balanced dataset with 16,000 total images**, improving generalization and reducing class bias.

---

## 🧠 Model Training

We fine-tuned a **ResNet-18** model from `torchvision.models` using the following settings:

- **Input size:** 224x224 (resized and normalized)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (learning rate = `1e-4`)
- **Epochs:** 10
- **Augmentations:** Online transforms during training
- **GPU Support:** Trained using **NVIDIA RTX 3050 Ti** with CUDA acceleration
- **Class weights:** Dynamically calculated based on training data distribution

### 🔍 Evaluation Metrics:
- Accuracy
- F1-Score (weighted)
- Confusion Matrix
- ROC Curves for all classes

---

## 🩺 Inference – Disease Prediction

A trained model (`best_model.pt`) can be used to predict the class of a new lesion image.
Model Link : https://www.kaggle.com/models/sametcansahin/skin_lesion_resnet18_szm/
### Example Code:

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

# Load and predict
image = Image.open("test_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()
