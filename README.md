🌾 Crop Disease Detection using Computer Vision

 bjective
This project focuses on developing a deep learning based image classification system that can automatically identify diseases in crop leaves using Computer Vision techniques. The system is designed to assist farmers in detecting plant health issues at an early stage, enabling faster decisions, reducing yield loss, and minimizing the overuse of chemical treatments. Model performance is measured using accuracy, precision, and recall, with special focus on reducing false negatives in disease detection.

🗂️ Dataset
Dataset Name: PlantVillage Dataset
Source: Kaggle (PlantVillage Dataset)
Dataset Link: [PlantVillage_Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

Consists of labeled leaf images covering both healthy and infected plant conditions
Spans across several crop varieties with numerous disease classifications


📁 Project Structure
Crop-Disease-Detection-CNN/
│
├── week1_eda.ipynb                     ← EDA, preprocessing pipeline
├── week2_ccn.ipynb                     ← Custom CNN training
├── week3_transfer_learning.ipynb       ← MobileNetV2 transfer learning
├── week4_confisionMatrix.ipynb         ← Confusion matrix & evaluation
├── week4_interface.ipynb               ← Inference script
│
├── app.py                              ← Flask web application backend
├── templates/
│   └── index.html                      ← Frontend UI (LeafScan)
│
├── confusion_matrix.png                ← Week 4 evaluation output
├── training_curves.png                 ← Week 2 training curves
├── transfer_learning_curves.png        ← Week 3 training curves
│
├── .gitignore                          ← Excludes large .pth model files
└── README.md

Week 1: Image Acquisition, EDA & Preprocessing
Data Loading

Accessed dataset from local system directory
Confirmed class-wise folder organization and image availability

Exploratory Data Analysis (EDA)

Displayed representative images from each available class
Examined image count per class to identify distribution imbalances

Data Preprocessing

Standardized all images to 224 × 224 resolution
Scaled pixel intensities from range 0–255 down to 0–1
Partitioned dataset into three separate subsets:

Train (70%)
Validation (15%)
Test (15%)

Tools & Technologies
Python PyTorch Torchvision NumPy Matplotlib PIL



Week 2: Data Augmentation & Custom CNN Architecture
Framework
PyTorch
Data Augmentation Pipeline
TechniqueDetailsRandom Horizontal FlipImproves spatial generalizationRandom Vertical FlipHandles varied orientationsRandom Rotation±30°Color JitterBrightness & contrast ±0.3Resize224 × 224NormalizationImageNet mean & std
Custom CNN Architecture
Layer BlockDetailsConv Block 1Conv2D(3→32) + ReLU + MaxPoolConv Block 2Conv2D(32→64) + ReLU + MaxPoolConv Block 3Conv2D(64→128) + ReLU + MaxPoolFully ConnectedLinear(128×28×28 → 512) + ReLU + Dropout(0.5) + Linear(512→38)
Training Configuration
ParameterValueEpochs10Batch Size32Learning Rate0.001Loss FunctionCrossEntropyLossOptimizerAdamEarly StoppingPatience = 3DeviceCUDA / CPU
Outputs

Training & Validation curves saved as training_curves.png
Best model saved as custom_cnn_plantvillage.pth

Tools & Technologies
Python PyTorch Torchvision Matplotlib



Week 3: Transfer Learning with MobileNetV2
Approach
Replaced custom CNN with MobileNetV2 pretrained on ImageNet to achieve higher accuracy through transfer learning.
Architecture
ComponentDetailsBase ModelMobileNetV2 (pretrained on ImageNet)Base LayersFrozen — used for feature extraction onlyClassifier HeadDropout(0.5) → Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→38)
Training Configuration
ParameterValueEpochs10Batch Size16Learning Rate0.0001Loss FunctionCrossEntropyLossOptimizerAdam (classifier layers only)SchedulerReduceLROnPlateau (patience=2, factor=0.5)Early StoppingPatience = 3DeviceCPU
Results
ModelTest AccuracyCustom CNN (Week 2)~70–75%MobileNetV2 (Week 3)~90%+
Outputs

Training curves saved as transfer_learning_curves.png
Best model saved as mobilenet_plantvillage.pth
Full classification report printed (Precision, Recall, F1 per class)

Tools & Technologies
Python PyTorch Torchvision scikit-learn Matplotlib



Week 4: Evaluation, Inference & Deployment
Confusion Matrix

Generated confusion matrix across all 38 classes on the test set
Identified most confused class pairs (visually similar diseases)
Saved as confusion_matrix.png

Inference Script

Takes any raw unseen leaf image as input
Loads trained MobileNetV2 weights
Outputs:

Predicted disease name
Confidence score (%)
Top 5 predictions



Flask Web Application — LeafScan
Built a complete web app connecting the trained model to a browser-based UI.
How to run:
bashpip install flask torch torchvision pillow scikit-learn
python app.py
Then open: http://localhost:5000
Features:

Drag and drop leaf image upload
Instant disease prediction with confidence score
Top 5 predictions displayed
Health status indicator (Healthy ✅ / Diseased ⚠️)
Treatment recommendation

Note on Model Weights
Model .pth files are excluded from the repository via .gitignore due to GitHub's 100MB file size limit (model is ~600MB). In production, weights would be hosted on cloud storage (Google Drive / AWS S3).
Tools & Technologies
Python PyTorch Flask HTML/CSS/JavaScript scikit-learn seaborn


🔑 Key Techniques Used

Data Augmentation — Random flips, rotations, color jitter
Transfer Learning — MobileNetV2 pretrained on ImageNet
Early Stopping — Prevents overfitting, saves best weights
LR Scheduling — ReduceLROnPlateau
Checkpointing — Resume training after interruption
Confusion Matrix — Identifies misclassification patterns
Web Deployment — Flask backend + HTML/CSS/JS frontend


Conclusion
This project delivers a reliable and efficient solution for automated crop disease identification using deep learning. The final MobileNetV2 model achieves 90%+ accuracy across 38 plant disease classes and is deployed as a web application accessible via any browser — contributing towards smarter agricultural practices and better food security outcomes.
