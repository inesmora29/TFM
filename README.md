# Evaluating the Efficacy of Vision Transformer Compared to Traditional Deep Learning Architectures for Diabetic Retinopathy Detection

## Overview
This repository contains the code and resources for the project **"Evaluating the Efficacy of Vision Transformer Compared to Traditional Deep Learning Architectures for Diabetic Retinopathy Detection"**. The project explores the performance of various deep learning models, including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), in classifying diabetic retinopathy (DR) from retinal fundus images.

## Project Structure
```
├── data/                     # Directory for datasets (Messidor, IDRiD)
├── models/                   # Saved trained models
├── notebooks/                # Jupyter notebooks for experimentation and visualization
├── src/                      # Source code for model training and evaluation
│   ├── preprocessing.py      # Data preprocessing scripts
│   ├── train.py              # Training script for models
│   ├── evaluate.py           # Model evaluation script
│   ├── models.py             # Model architectures (CNN, ResNet50, VGG16, ViT-B/16)
│   ├── config.yaml           # Configuration file for hyperparameters and settings
│   ├── cli.py                # Command-line interface for running scripts
├── results/                  # Evaluation results and reports
├── requirements.txt          # List of dependencies
├── Dockerfile                # Docker setup for easy execution
├── run_colab.ipynb           # Colab notebook for easy usage
├── README.md                 # Project documentation (this file)
└── LICENSE                   # License file
```

## Installation
To set up the environment and run the code, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/inesmora29/diabetic-retinopathy-detection.git
   cd diabetic-retinopathy-detection
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code
### **Using CLI**
To preprocess the data, train a model, and evaluate results, run:
```bash
python src/cli.py --task preprocess
python src/cli.py --task train --model vit --epochs 50 --batch_size 32
python src/cli.py --task evaluate --model vit
```
Replace `vit` with `cnn`, `resnet50`, or `vgg16` to train different models.

### **Using Docker**
Build and run the container:
```bash
docker build -t dr-detection .
docker run --rm dr-detection --task train --model vit
```

### **Using Google Colab**
Open the provided **`run_colab.ipynb`** notebook for execution in the cloud.

## GPU (CUDA) Support
This project supports **GPU acceleration** using CUDA for PyTorch and TensorFlow. Follow these steps to enable GPU usage:

1. **Check CUDA Version**:
   ```bash
   nvcc --version
   ```
   Example output:
   ```
   nvcc: NVIDIA (R) Cuda compiler
   release 11.8, V11.8.89
   ```
2. **Install CUDA-Compatible Dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install tensorflow tensorflow-gpu
   ```
3. **Verify GPU Usage**:
   ```python
   import torch
   import tensorflow as tf
   print("PyTorch GPU:", torch.cuda.is_available())
   print("TensorFlow GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
   ```
If you don’t have a GPU, install the CPU-only version:
```bash
pip install torch torchvision torchaudio
pip install tensorflow
```

## Dataset
This project uses the **Messidor** and **IDRiD** datasets for training and evaluation. Due to licensing restrictions, the datasets are not included in this repository. You can download them from their official sources and place them in the `data/` directory.

## Results
- The **ViT-B/16** model achieved the highest accuracy of **77.78%**, outperforming CNN-based models.
- Preprocessing techniques (e.g., CLAHE) showed inconsistent effects on model performance.
- Transfer learning with pretrained architectures improved accuracy over a custom CNN model.

## Future Work
- Experiment with ensemble learning to combine CNNs and ViTs.
- Optimize preprocessing techniques for improved feature extraction.
- Expand evaluation on different DR severity levels.

## License
This project is licensed under the [Creative Commons BY-NC 3.0 License](https://creativecommons.org/licenses/by-nc/3.0).

## Acknowledgments
- **Author:** Inés del Carmen Mora García
- **Tutor:** Alfredo Madrid García
- **SRP Supervisor:** Dra. Agnès Perez Millan
- **Program:** Màster en Bioinformàtica i Bioestadística

