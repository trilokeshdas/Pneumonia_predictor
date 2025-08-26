# 🩺 Pneumonia Detection from Chest X-Rays  

A **Streamlit web application** that uses a deep learning model to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.  
The app predicts probabilities for both classes and provides an easy-to-understand health status message.  

---

## 📌 Description  
Pneumonia is a serious lung infection, and early detection can save lives. This project leverages **deep learning (CNN)** to automatically analyze chest X-ray images and determine whether they are normal or show signs of pneumonia.  
The web interface is built with **Streamlit**, making it simple and interactive for users.  

---

## 🚀 Features  
- Upload chest X-ray images (`.jpg`, `.jpeg`, `.png`).  
- Predicts whether the lungs are **Normal** or **Pneumonia** affected.  
- Shows **confidence scores** for each class.  
- Provides a **patient status conclusion** (Low/Moderate/High risk).  
- Simple and interactive **Streamlit interface**.  

---

## 🛠 Tech Stack  
- **Python 3.x**  
- **TensorFlow / Keras** – deep learning model  
- **Streamlit** – web app interface  
- **NumPy** – numerical computations  
- **Pillow (PIL)** – image preprocessing  

---

## ⚙️ Installation  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection
