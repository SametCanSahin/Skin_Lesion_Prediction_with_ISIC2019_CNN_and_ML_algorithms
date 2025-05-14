import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------- GPU Kontrolü ----------
print("CUDA available:", torch.cuda.is_available())
print("GPU sayısı:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ---------- Dizin--------------
root_dir = r'C:\Users\Samet\Desktop\data'
image_dir = os.path.join(root_dir, r'Augmented_Balanced_V2')

# ---------- Dataset Sınıfı ----------
class LesionDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'SCC': 6, 'VASC': 7}

        for cls in os.listdir(image_root):
            cls_folder = os.path.join(image_root, cls)
            if os.path.isdir(cls_folder) and cls in self.class_to_idx:
                for img_name in os.listdir(cls_folder):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- Dataset ve Split ----------
dataset = LesionDataset(image_dir, transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------- Model ----------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 8)  # 8 sınıf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    print(f"Multiple GPUs detected: Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)
print("Model running on:", next(model.parameters()).device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------- Eğitim ----------
num_epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_accuracies.append(100 * train_correct / train_total)
    val_accuracies.append(100 * val_correct / val_total)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Val Loss: {val_loss/len(val_loader):.4f} "
          f"Train Acc: {100*train_correct/train_total:.2f}% "
          f"Val Acc: {100*val_correct/val_total:.2f}%")

# --------------------
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# ---------- Metrikler ----------
class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
print(f"Weighted F1 Score: {f1*100:.2f}%\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
# model eğitimi sonuna ekle
torch.save(model.state_dict(), "best_segmented_sharpened.pt")

# ---------- Confusion Matrix ----------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ---------- ROC Curve ----------
n_classes = 8
fpr, tpr, roc_auc = {}, {}, {}
labels_np = np.array(all_labels)
probs_np = np.array(all_probs)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((labels_np == i).astype(int), probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,8))
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, color=colors[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# ---------- Sınıf Dağılımı ----------
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]
class_distribution_train = Counter(train_labels)
class_distribution_test = Counter(test_labels)

print("Train Dataset Sınıf Dağılımı:")
for class_idx, count in class_distribution_train.items():
    print(f"{class_names[class_idx]}: {count} örnek")

print("\nTest Dataset Sınıf Dağılımı:")
for class_idx, count in class_distribution_test.items():
    print(f"{class_names[class_idx]}: {count} örnek")
