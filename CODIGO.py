from google.colab import drive
drive.mount('/content/drive')

from collections import Counter
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch. utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import seaborn as sns

### CREATION OF THE DATASETS ###

# carpetas con las imagenes y archivos excel con diagnosticos
folder_paths = [
    '/content/drive/MyDrive/Base11', '/content/drive/MyDrive/Base12', '/content/drive/MyDrive/Base13',
    '/content/drive/MyDrive/Base14', '/content/drive/MyDrive/Base21', '/content/drive/MyDrive/Base22',
    '/content/drive/MyDrive/Base23', '/content/drive/MyDrive/Base24', '/content/drive/MyDrive/Base31',
    '/content/drive/MyDrive/Base32', '/content/drive/MyDrive/Base33', '/content/drive/MyDrive/Base34',
    '/content/drive/MyDrive/IDRID'
]

# funcion para extraer los diagnosticos
def get_diagnostics(folder_paths):
  diagnostics = {}
  for folder in folder_paths:
      excel_files = [f for f in os.listdir(folder) if f.endswith('.xlsx') or f.endswith('.xls')]
      for excel_file in excel_files:
          file_path = os.path.join(folder, excel_file)
          df = pd.read_excel(file_path)
          for _, row in df.iterrows():
              image_name = row['Image name']
              retinopathy_grade = row['Retinopathy grade']
              diagnostics[image_name] = 1 if retinopathy_grade > 0 else 0
  return diagnostics


diagnostics = get_diagnostics(folder_paths)
