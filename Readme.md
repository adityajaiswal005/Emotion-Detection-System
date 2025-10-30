# 🎭 Emotion Detection System using Machine Learning

This is a **Text-based Emotion Detection System** built using **Python, NLP (Natural Language Processing)**, and **Machine Learning**.  
It detects emotions like **joy, sadness, anger, fear, love, and surprise** from any given text input and visualizes prediction confidence through a **bar chart** in a **Streamlit web app**.


---

## 🧠 Overview

The Emotion Detection System uses **TF-IDF Vectorization** and a **Naive Bayes classifier** to predict human emotions from text.  
It includes:
- A machine learning backend model
- A clean Streamlit-based user interface
- Bar chart visualization of emotion confidence (ranging from 0 to 1)

---

## ✨ Features

✅ Predicts emotions from text input  
✅ Clean and simple Streamlit UI  
✅ Static Matplotlib bar chart showing confidence scores  
✅ High accuracy text classification using NLP  
✅ Pre-trained model and vectorizer are reusable (`.pkl` files)  
✅ Customizable for your own dataset  

---

## 🧰 Technologies Used

| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| ML Algorithm | Multinomial Naive Bayes |
| Feature Extraction | TF-IDF Vectorizer |
| Libraries | Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit |
| Dataset | Custom CSV / Hugging Face `dair-ai/emotion` |

---

## 📂 Dataset

You need a CSV file named **`emotion_detection.csv`** containing at least two columns:

| text | emotion |
|------|----------|
| I am so happy today! | joy |
| This is terrible... | sadness |
| I love you! | love |
| That scared me a lot | fear |


# 1️⃣ Clone this repository
git clone https://github.com/your-username/Emotion-Detection.git
cd Emotion-Detection

# 2️⃣ Install dependencies

pip install streamlit pandas numpy scikit-learn matplotlib 



# 3️⃣ Train the model
python model.py

# 4️⃣ Run the Streamlit app
streamlit run app.py

