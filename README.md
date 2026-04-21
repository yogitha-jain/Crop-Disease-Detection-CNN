🌾 Crop Disease Detection using Computer Vision
📌 Project Status: Ongoing Internship Project – Week 1 Completed

🎯 Objective
This project focuses on developing a deep learning based image classification system that can automatically identify diseases in crop leaves using Computer Vision techniques. The system is designed to assist farmers in detecting plant health issues at an early stage, enabling faster decisions, reducing yield loss, and minimizing the overuse of chemical treatments. Model performance is measured using accuracy, precision, and recall, with special focus on reducing false negatives in disease detection.

🗂️ Dataset
Dataset Name: PlantVillage Dataset
Source: Kaggle (PlantVillage Dataset)
Dataset Link: [PlantVillage_Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Consists of labeled leaf images covering both healthy and infected plant conditions
Spans across several crop varieties with numerous disease classifications


📊 Week 1: Image Acquisition, EDA & Preprocessing
✅ Data Loading

Accessed dataset from local system directory
Confirmed class-wise folder organization and image availability

✅ Exploratory Data Analysis (EDA)

Displayed representative images from each available class
Examined image count per class to identify distribution imbalances

✅ Data Preprocessing

Standardized all images to 224 × 224 resolution
Scaled pixel intensities from range 0–255 down to 0–1
Partitioned dataset into three separate subsets:

Train (70%)
Validation (15%)
Test (15%)

⚙️ Preprocessing Pipeline

Systematic dataset partitioning using image generators
Pipeline built using TensorFlow for efficient data loading
Pixel normalization and batch prefetching applied for optimized performance


🛠️ Tools & Technologies

Python
TensorFlow / Keras
NumPy
Matplotlib


📁 Project Structure
Crop-Disease-Detection/
|── PlantVillage_color/
|── PlantVillage_split/
|   |── train/
|   |── val/
|   └── test/
|── notebooks/
|── README.md

💡 Conclusion
This project aims to deliver a reliable and efficient solution for automated crop disease identification, contributing towards smarter agricultural practices and better food security outcomes.
