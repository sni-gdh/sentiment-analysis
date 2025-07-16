# 🧠 Sentiment Analysis Using CNN and Facial Expression Recognition

## 📌 Project Overview

This project implements a **Sentiment Analysis system** that detects human emotions based on facial expressions using **Convolutional Neural Networks (CNNs)**. The model is trained on the **Fer2013 dataset**, which contains labeled facial images representing various emotions like happiness, sadness, anger, surprise, etc.

The system uses **OpenCV** for face detection, **Keras** and **TensorFlow** for model development, and is wrapped in a **Jupyter Notebook** for step-by-step demonstration.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- Matplotlib
- NumPy
- Fer2013 Dataset

---

## 🧑‍🎓 Features

- Face detection using Haar Cascades (OpenCV)
- Preprocessing of facial image data
- Emotion classification using CNN
- Real-time emotion detection from webcam *(optional)*
- Visualization of training progress (loss/accuracy plots)

---

## 📂 Project Structure
<pre> ```bash
├── dataset/
│ └── fer2013.csv # Facial emotion data
├── models/
│ └── emotion_model.h5 # Trained CNN model
├── notebooks/
│ └── sentiment_analysis.ipynb # Main Jupyter Notebook
├── utils/
│ └── preprocess.py # Preprocessing helpers
├── README.md
└── requirements.txt ``` </pre>

---

---

## 🚀 Getting Started

### 🔄 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-cnn.git
cd sentiment-analysis-cnn
```
### 📦 2. Install Dependencies
pip install -r requirements.txt

### ▶️ 3. Run the Notebook
jupyter notebook notebooks/sentiment_analysis.ipynb

## 📊 Dataset
The model uses the Fer2013 Dataset. It contains 48x48 grayscale facial expression images.
Download and place it under the dataset/ directory.

## 🎯 Results
✅ Achieved ~90% accuracy on validation data
✅ Model generalizes well on real-world images

## 🤝 Contributing
Contributions are welcome!
Feel free to open an issue or submit a pull request to enhance this project.




