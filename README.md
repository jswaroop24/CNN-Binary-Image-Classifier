# CNN Binary Image Classifier

## 📌 Problem Statement
The goal of this project is to build a **Convolutional Neural Network (CNN)** to perform **binary image classification**.  
Given an input image, the model predicts which of the two classes the image belongs to.

---

## 🧠 Model Overview
This project uses a **deep learning-based CNN architecture** implemented using **TensorFlow and Keras**.

### Architecture Details:
- 2 Convolutional Layers
- MaxPooling layers
- Flatten layer
- 2 Fully Connected (Dense) layers
- Dropout layers to reduce overfitting
- Sigmoid activation for binary classification

---

## 📂 Dataset
- Images are split into **training** and **testing** sets
- Images are resized and normalized during preprocessing
- Binary labels are used

*(Dataset path is loaded locally in the notebook)*

---

## ⚙️ Data Preprocessing
- Image resizing
- Normalization (pixel values scaled)
- Train-test split
- Batch processing using data generators

---

## 🚀 Training Details
- Optimizer: SGD
- Loss Function: Binary Cross entropy
- Metrics: Accuracy, Loss
- Epoch-based training with validation

---

## 📊 Results
- Model performance is evaluated using accuracy and loss curves
- Training and validation metrics are visualized
- Training Accuracy at `96%` while Validation Accuracy at `80%`

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## ▶️ How to Run the Project

1.  **Clone the repository:**
```bash
git clone https://github.com/USERNAME/cnn-binary-image-classifier.git
cd CNN-Binary-Image-Classifier
```

2.  **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook main.ipynb
```

---

## ✅ Notes

- Recommended Python version: 3.8 – 3.11
- GPU support is optional but recommended for faster training

---

## 📌 Future Improvements

- Hyperparameter tuning

- Data augmentation

- Transfer learning (VGG16, ResNet)

- Model saving and deployment
