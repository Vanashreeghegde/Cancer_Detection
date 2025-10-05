# 🩺 Cancer Detection with PyTorch  

This project implements a deep learning pipeline using **PyTorch** to classify medical images for metastatic cancer detection. The dataset used is a subset of the **PatchCamelyon (PCam) dataset**, which contains histopathologic scans of lymph node sections.  

## 📌 Project Overview  
- Built a **Convolutional Neural Network (CNN)** using **ResNet34 (pretrained)** for binary classification (cancer vs. non-cancer).  
- Trained and validated the model on a sample of PCam dataset.  
- Applied **data augmentation & normalization** techniques to improve generalization.  
- Evaluated model performance using **accuracy metrics**.  

⚠️ **Note**: Due to resource limitations in the lab environment, the model was trained on a small subset (few epochs) and achieved ~50% accuracy. With more data & training epochs, the model performance can be significantly improved.  

---

## ⚙️ Tech Stack  
- **Language**: Python  
- **Libraries**: PyTorch, TorchVision, NumPy, Pandas, Matplotlib, Seaborn  
- **Model**: ResNet34 (pretrained on ImageNet, fine-tuned for binary classification)  
- **Dataset**: [PatchCamelyon (PCam)](https://github.com/basveeling/pcam)  

---

## 📂 Dataset  
The PCam dataset contains color images (96 x 96 px) extracted from histopathologic scans. Each image is labeled:  
- **0** → Normal tissue  
- **1** → Metastatic tissue  

In this project, we used a **sample dataset** provided by IBM Skills Network Labs.  

---

## 🚀 Project Workflow  
1. **Setup & Install Dependencies**  
   - Installed and imported required libraries.  
   - Verified GPU availability.  

2. **Data Preparation**  
   - Downloaded and extracted dataset (`labels.csv` & `data_sample.zip`).  
   - Created custom PyTorch `Dataset` class.  
   - Applied transformations (resize, normalization, random flips, rotations).  
   - Split dataset into **train, validation, test** (70/15/15).  

3. **Model**  
   - Used **ResNet34 (pretrained=True)**.  
   - Replaced final fully-connected layer with 2 outputs (binary classification).  
   - Optimizer: **Adam**  
   - Loss Function: **CrossEntropyLoss**  

4. **Training**  
   - Trained with batch size = 30, learning rate = 3e-4, epochs = 1 (for demo).  

5. **Evaluation**  
   - Tested model on test dataset.  
   - Accuracy achieved: **50%**  

---

## 📊 Results  
- Achieved ~50% accuracy on test set (limited training).  
- Visualized random dataset samples with labels.  
- Framework allows **scaling to full dataset & more epochs** for improved performance.  

---

## 🔮 Future Improvements  
- Train with full PCam dataset.  
- Increase number of epochs for better learning.  
- Experiment with different architectures (ResNet50, EfficientNet).  
- Apply advanced augmentation and transfer learning.  

---

## 🏅 Certificate  
This project was completed as part of the **IBM Skills Network - Deep Learning & AI Training**.  
Certificate: _(Upload certificate image/PDF in repo and link it here)_.  

---

## 👨‍💻 Author  
- **Vanashree G. Hegde**  

---

## 📜 License  
This project is based on IBM Skills Network materials and PCam dataset (CC0 License).  
