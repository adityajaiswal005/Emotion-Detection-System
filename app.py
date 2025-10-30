import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend (important for Streamlit)
import matplotlib.pyplot as plt

# 🎯 Load Model & Vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Emotion Detection App", page_icon="🎭", layout="centered")

st.title("🎭 Emotion Detection System")


st.write("Type any sentence below and find out the **emotion** behind it ")

# 🧾 Input Text
user_input = st.text_area("📝 Enter your text here:")

# 🧮 Predict Emotion
if st.button("🔍 Detect Emotion"):
    if user_input.strip():
        # Transform input
        vec = vectorizer.transform([user_input])

        # Predict probabilities for all classes
        probabilities = model.predict_proba(vec)[0]
        labels = model.classes_
        prediction = labels[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        # 🧠 Show Result
        st.markdown(f"### 🧠 Predicted Emotion: **{prediction.upper()}**")
        st.progress(confidence)
        st.write(f"Confidence Level: **{confidence:.3f}**")

        # Create DataFrame for clarity
        prob_df = pd.DataFrame({
            "Emotion": labels,
            "Confidence": probabilities
        }).sort_values(by="Confidence", ascending=False)

        # 🎨 Matplotlib Static Bar Chart
        st.subheader("📊 Emotion Confidence (Static Chart)")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(prob_df["Emotion"], prob_df["Confidence"], color="cornflowerblue", edgecolor="black")
        ax.set_title("Emotion Confidence Levels (0–1 Range)", fontsize=13, pad=10)
        ax.set_xlabel("Emotion Class", fontsize=11)
        ax.set_ylabel("Confidence", fontsize=11)
        ax.set_ylim(0, 1)  # ✅ range from 0 to 1
        plt.xticks(rotation=25)
        st.pyplot(fig)

        # Optional — Display data table
        st.write(prob_df.style.background_gradient(cmap="Blues", axis=0))

    else:
        st.warning("⚠️ Please enter some text before clicking Detect!")

st.markdown("---")
st.caption("💡 Tip: Try sentences like *'I am so happy today!'* or *'This makes me sad.'*")
