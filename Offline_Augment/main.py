import os
import cv2
import random
import albumentations as A
from tqdm import tqdm
import numpy as np

# ---------- Ayarlar ----------
input_dir = r"C:\Users\Samet\Desktop\data\ISIC_2019_Merged"             # Girdi: sınıfların alt klasör olarak bulunduğu yer
output_dir = r"C:\Users\Samet\Desktop\data\Augmented_Balanced_V2"        # Çıktı: dengelenmiş veri buraya
target_per_class = 2000

# ---------- Augmentasyonlar ----------
transform = A.Compose([
    A.RandomScale(scale_limit=0.2, p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), per_channel=False, p=1.0),
    A.Sharpen(alpha=(0.3, 0.6), lightness=(0.95, 1.0), p=1.0)
])


# ---------- Yardımcı Fonksiyon ----------
def save_augmented(image, save_path, count):
    augmented = transform(image=image)['image']
    out_path = os.path.join(save_path, f"aug_{count}.jpg")
    cv2.imwrite(out_path, augmented)

# ---------- Her sınıf için işlemler----------
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    original_count = len(image_files)

    print(f"\n▶ Sınıf: {class_name} | Orijinal: {original_count}")

    # Rastgele 2000 seçim (NV ve MEL)
    if class_name in ['NV', 'MEL']:
        selected = image_files[:target_per_class]
        for i, img_name in enumerate(selected):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                cv2.imwrite(os.path.join(output_class_dir, f"{i}_{class_name.lower()}.jpg"), image)

    else:
        # Orijinal verileri kopyala
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                cv2.imwrite(os.path.join(output_class_dir, f"orig_{i}.jpg"), image)

        # Eksikse augment et
        needed = target_per_class - original_count
        print(f"  ↪ Eksik: {needed} → augment ediliyor...")

        count = 0
        while count < needed:
            src = random.choice(image_files)
            src_path = os.path.join(class_path, src)
            image = cv2.imread(src_path)
            if image is not None:
                save_augmented(image, output_class_dir, count)
                count += 1
