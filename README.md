# Crop-Disease-Detection-CNN

## Project Overview
Global agriculture faces massive challenges as plant diseases threaten 
food security and crop yields. This project develops a deep learning 
based image classifier to detect early signs of disease from plant 
leaf images, enabling timely and targeted interventions.

## Dataset
- PlantVillage Dataset (54,305 images)
- 38 classes (healthy + diseased leaves)
- Plants: Tomato, Apple, Corn, Grape, Potato and more

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV / PIL
- Matplotlib / Seaborn

## Project Roadmap
- Week 1 — Data Acquisition, EDA & Preprocessing ✅
- Week 2 — Data Augmentation & Custom CNN 🔄
- Week 3 — Transfer Learning (ResNet50 / EfficientNet) ⏳
- Week 4 — Evaluation & Metrics ⏳

## Week 1 — Completed
- Loaded PlantVillage dataset (38 classes, 54k images)
- Performed Exploratory Data Analysis (EDA)
- Plotted class distribution and sample images
- Built preprocessing pipeline (224x224, normalized)
- Split data into Train (70%) / Validation (15%) / Test (15%)
