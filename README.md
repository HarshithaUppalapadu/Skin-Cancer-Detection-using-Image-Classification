# 🧠 Skin Cancer Detection using Image Classification

This project uses **Convolutional Neural Networks (CNNs)** to classify skin lesion images and detect skin cancer types using the **HAM10000** dataset.  
It also includes a **Tkinter-based GUI** that allows users to upload an image, select a trained model, and view classification results interactively.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [GUI Application](#gui-application)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Results](#results)
- [Author](#author)

---

## 🩺 Overview

Skin cancer is one of the most common cancers globally, and early detection can greatly improve survival rates.  
This project builds a **deep learning-based classifier** using CNNs to identify seven types of skin lesions, wrapped in an intuitive **Tkinter GUI** for easy use.

---

## 🌟 Features

- ✅ Trains a CNN model on HAM10000 skin lesion dataset  
- ✅ Preprocesses and standardizes dermatoscopic images  
- ✅ Performs one-hot encoding of diagnostic categories  
- ✅ GUI interface to:
  - Upload and view an image  
  - Select between two trained CNN models  
  - Display lesion metadata  
  - Classify and show prediction probabilities  
- ✅ Option to save classification results to a text file  

---

## 🧾 Dataset

**Dataset:** [HAM10000 – Human Against Machine with 10000 training images](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

The dataset contains:
- **10,015** dermatoscopic images of skin lesions  
- **7 diagnostic categories** (melanocytic nevi, melanoma, benign keratosis, etc.)  
- Metadata file: `HAM10000_metadata.csv`

Each image is resized to **100×75 pixels** before training.

---

## 🧠 Model Architecture

The CNN architecture includes:

| Layer | Description |
|-------|--------------|
| Conv2D + ReLU | Feature extraction |
| Conv2D + ReLU | Deeper feature extraction |
| MaxPooling2D | Reduces spatial dimensions |
| Dropout | Prevents overfitting |
| Conv2D + ReLU (x2) | Further feature learning |
| MaxPooling2D | Pooling layer |
| Dropout | Regularization |
| Flatten | Converts feature maps into a 1D vector |
| Dense (128) + ReLU | Fully connected layer |
| Dropout | Regularization |
| Dense (7) + Softmax | Output layer for 7 cancer types |

**Optimizer:** Adam (`lr = 0.0001`)  
**Loss:** Categorical Cross-Entropy  
**Epochs:** 5  
**Batch Size:** 25  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/SkinCancerClassification.git
cd SkinCancerClassification
