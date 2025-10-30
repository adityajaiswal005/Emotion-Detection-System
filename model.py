# model.py
import pandas as pd
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1️⃣ Load dataset
data = pd.read_csv("emotion_detection.csv", encoding='utf-8', on_bad_lines='skip')


# 2️⃣ Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(clean_text)

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['emotion'], test_size=0.2, random_state=42
)

# 4️⃣ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Model
model = MultinomialNB()
print("Training data size:", X_train_vec.shape)
print("Training model... please wait")
model.fit(X_train_vec, y_train)
print("Training complete ✅")


# 6️⃣ Save Model & Vectorizer
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
print(data['emotion'].value_counts())

