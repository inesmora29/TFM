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
├── results/                  # Evaluation results and reports
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation (this file)
└── LICENSE                   # License file
```

## Installation
To set up the environment and run the code, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/diabetic-retinopathy-detection.git
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

## Dataset
This project uses the **Messidor** and **IDRiD** datasets for training and evaluation. Due to licensing restrictions, the datasets are not included in this repository. You can download them from their official sources and place them in the `data/` directory.

## Training the Models
To train a model, run:
```bash
python src/train.py --model vit --epochs 50 --batch_size 32
```
Replace `vit` with `cnn`, `resnet50`, or `vgg16` to train different models.

## Evaluating the Models
After training, evaluate the models using:
```bash
python src/evaluate.py --model vit --dataset test
```

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

