# 🧠 Handwritten Digit Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) using the MNIST dataset. The model achieves over **99% accuracy** and can also predict real-world digit images.

---

## 📂 Files in the Repository

- `main.py` – Script to load the trained model and predict digits from external images
- `cnn_digit_classifier.h5` – Trained CNN model (Keras format)
- `Digit_recognition_CNN.ipynb` – Full Jupyter Notebook for data preprocessing, model training, and evaluation
- `README.md` – Project documentation
- *(Optional)* `my_digit.png` – A real handwritten digit image to test the model

---

## 🧠 Model Summary

- **Architecture:**  
  - 2 × Conv2D + ReLU + MaxPooling
  - Flatten → Dense(128) + ReLU
  - Dropout(0.5) → Dense(10) + Softmax
- **Trained on:** MNIST dataset (60,000 training, 10,000 testing samples)
- **Final accuracy:** 99.0% on test set

### 🔍 Classification Report
