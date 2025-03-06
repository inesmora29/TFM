import os
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Dataset paths
FOLDER_PATHS = ["data/Base11", "data/Base12", "data/Base13", "data/Base14", "data/IDRID"]
INPUT_SIZE = (224, 224)

def get_diagnostics(folder_paths):
    diagnostics = {}
    for folder in folder_paths:
        excel_files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]
        for file in excel_files:
            df = pd.read_excel(os.path.join(folder, file))
            for _, row in df.iterrows():
                diagnostics[row['Image name']] = 1 if row['Retinopathy grade'] > 0 else 0
    return diagnostics

def enhance_image(image):
    """Applies CLAHE to enhance contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = enhance_image(image)
    image = cv2.resize(image, INPUT_SIZE)
    return np.transpose(image, (2, 0, 1))  # Convert to PyTorch format

def prepare_dataset():
    diagnostics = get_diagnostics(FOLDER_PATHS)
    data, labels = [], []

    for folder in FOLDER_PATHS:
        for image_name, label in diagnostics.items():
            image_path = os.path.join(folder, image_name)
            if os.path.exists(image_path):
                data.append(preprocess_image(image_path))
                labels.append(label)

    np.save("data/X.npy", np.array(data))
    np.save("data/y.npy", np.array(labels))
    print("Preprocessed data saved.")

if __name__ == "__main__":
    prepare_dataset()
