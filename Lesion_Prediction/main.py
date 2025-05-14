import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------- Sınıf Etiketleri ----------
class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']

# ---------- Model ve Cihaz Ayarları ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Uyarı çözümü: pretrained yerine weights kullan
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model_path = r'C:\Users\Samet\Downloads\best_model.pt'

# 📌 Eğer eğitim sırasında DataParallel kullanıldıysa "module." prefix'lerini kaldır
state_dict = torch.load(model_path, map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ---------- Transform (augmentasyonsuz, test için) ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Tahmin Fonksiyonu ----------
def predict_image(image_path):
    assert os.path.exists(image_path), f"Dosya bulunamadı: {image_path}"

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_idx].item()

    print(f"\n✅ Tahmin: {class_names[predicted_idx]} ({confidence*100:.2f}% güven)\n")

    plt.imshow(image)
    plt.title(f"Prediction: {class_names[predicted_idx]}\nConfidence: {confidence*100:.2f}%")
    plt.axis('off')
    plt.show()

# ---------- Kullanım ----------
if __name__ == '__main__':
    image_path = r'C:\Users\Samet\Desktop\WhatsApp Görsel 2025-05-14 saat 01.16.31_77aa6c5b.jpg'
    predict_image(image_path)
