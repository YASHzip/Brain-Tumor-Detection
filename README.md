# 🧠 Brain Tumor Detection using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.8+-blue)  
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)  
![Status](https://img.shields.io/badge/Status-Active-brightgreen)  

---

## 📌 Project Overview  
This project focuses on **Brain Tumor Detection from MRI scans** using **Deep Learning (PyTorch)**.  
The trained model (`BrainTumorRecognition.pth`) can classify MRI images into **Tumor** and **No Tumor** categories.  

By automating tumor detection, this project supports **faster diagnosis** and could be extended for real-world medical use cases.  

---

## 📂 Project Structure
```bash
Brain-Tumor-Detection/
│── data/ # Dataset (not included in repo)
│ ├── Training/ # Training images
│ └── Testing/ # Testing images
│
│── BrainTumorRecognition.pth # Saved trained model
│── brain-tumor-mri-dataset.zip # Dataset archive (ignored in .gitignore)
│── datasetDownload.py # Script to download dataset
│── detectionFinal.py # Final prediction script
│── eval.py # Model evaluation script
│── train.py # Training script
│── requirements.txt # Dependencies
│── .gitignore # Git ignore file
```

---

## ⚙️ Installation  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```
### Install the dependencies
```bash
pip install -r requirements.txt

```
### Download the dataset (if you wish to train the model yourself)
```bash
python datasetDownload.py
```
---
## 🚀 Usage
### Train the model (if you wish to retrain)
```bash
python train.py
```
### Run predictions on test images
```bash
python detectionFinal.py
```
### Evaluate the model performance
```bash
python eval.py
```

---

## 📊 Results
✅ High accuracy on test dataset
✅ Detects brain tumor vs no tumor from MRI scans

## 🛠️ Built With
- Python 🐍

- PyTorch 🔥

- Matplotlib 📈

---

## 📌 Future Improvements

- Improve accuracy with larger dataset

- Deploy as a Flask/Streamlit web app

---
## 🙏 Credits

This project is based on the Brain Tumor Detection tutorial by [NeuralNine](https://www.youtube.com/c/NeuralNine).  

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork this repo and improve the project.

---
✨ If you found this project helpful, don’t forget to star ⭐ the repo!
