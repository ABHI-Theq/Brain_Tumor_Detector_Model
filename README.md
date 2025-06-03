# 🧠 Brain Tumour Detector

This project focuses on building a deep learning model to classify MRI images for brain tumour detection. It involves two main approaches:

1. **Custom Convolutional Neural Network (CNN)** – A model built from scratch to explore the architecture design and training.
2. **Transfer Learning** – Utilizes pre-trained models (like VGG16, ResNet, or MobileNet) for faster training and better generalization on limited data.

> **Note**: The custom CNN model is not included in this repository due to its large size. However, the transfer learning model and code are available.

---

## 📁 Project Structure

```
BrainTumourDetector/
├── BrainTumourDetector.ipynb  # Jupyter notebook with code, models, training & evaluation
├── datasets/                  # Image dataset (assumed, not uploaded here)
└── README.md                  # Project overview and instructions
```

---

## 🧪 Models Overview

### 1. Custom CNN

* Built from scratch using Keras.
* Includes multiple Conv2D, MaxPooling2D, and Dense layers.
* Achieved decent performance on validation data.
* **Not uploaded** due to large model size.

### 2. Transfer Learning

* Based on a pre-trained model (e.g., `VGG16` or `MobileNet`).
* Final layers were customized to suit binary/multiclass classification.
* Fine-tuned for better domain-specific performance.
* **Included in this repository.**

---

## 🖼️ Dataset

* MRI images classified into two categories: `Tumour` and `No Tumour`.
* Images are resized and normalized before feeding into the models.
* Data augmentation is used to improve generalization.

---

## ⚙️ Dependencies

* Python 3.8+
* TensorFlow / Keras
* NumPy, Matplotlib, OpenCV
* Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/BrainTumourDetector.git
   cd BrainTumourDetector
   ```

2. Open the notebook:

   ```bash
   jupyter notebook BrainTumourDetector.ipynb
   ```

3. Run each cell step-by-step to:

   * Load data
   * Preprocess images
   * Train the model
   * Evaluate and visualize results

---

## 📊 Results

The transfer learning model achieved high accuracy and precision on the test set. It outperformed the custom model in terms of training speed and generalization.

---

## 📝 Notes

* You can train the model on Google Colab for better performance with GPU.
* You may customize the architecture or try different pre-trained models.
* This project can be extended to include segmentation for tumour localization.

---

## 📌 To Do

* [ ] Add inference script for live prediction
* [ ] Integrate with Flask/Django for web deployment
* [ ] Add model evaluation metrics like ROC-AUC, F1-Score

---

## 📬 Contact

For any questions or collaborations, feel free to reach out via \[your email] or GitHub Issues.

---

## 🧠 Acknowledgments

Special thanks to open-source contributors and datasets used in this project.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
