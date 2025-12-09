import streamlit as st
import pickle

# -------------------------------
# Load model + TF-IDF vectorizer
# -------------------------------
model = pickle.load(open("models/final_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))

# -------------------------------
# Streamlit UI
# -------------------------------
# Small title

# ----- Custom CSS -----
st.markdown("""
    <style>
        h1 { font-size: 1.8rem !important; }
        p { font-size: 0.9rem !important; }
        .stTextArea textarea { font-size: 0.9rem !important; }
        .stButton>button {
            font-size: 0.9rem !important;
            padding: 4px 10px;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will tell you whether it's **Positive** or **Negative**.")

review = st.text_area("Write your movie review here:", height=180)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please type a review first!")
    else:
        vec = tfidf.transform([review])
        prediction = model.predict(vec)[0]

        if prediction == "positive" or prediction == 1:
            st.success("🌟 Sentiment: **POSITIVE**")
        else:
            st.error("👎 Sentiment: **NEGATIVE**")
