# movie-review-sentiment-analysis-project


📘 Movie Review Sentiment Analysis

Using Logistic Regression, SVM, and KNN

📌 Project Overview

This project performs sentiment analysis on IMDb movie reviews to classify whether a review is positive or negative.
The goal is to build a machine learning pipeline that:

Cleans raw text

Converts text into numerical form using TF-IDF

Trains multiple ML models

Compares performance

Identifies the best classifier

📂 Dataset

We use the IMDb 50K Movie Reviews Dataset, containing:

50,000 reviews

Balanced labels: 25K positive, 25K negative

Two columns:

review → raw text

sentiment → positive / negative

After preprocessing, we generate:

clean_review → cleaned, normalized text

sentiment → encoded as 1 (positive), 0 (negative)

🧹 Data Preprocessing

Text cleaning includes:

✔ Removing HTML tags (<br/>)
✔ Removing special characters
✔ Lowercasing all text
✔ Removing stopwords (custom list, no NLTK required)
✔ Tokenization
✔ Joining cleaned text back

Final cleaned dataset is saved to:

dataset/cleaned/cleaned_reviews.csv

🔧 Feature Engineering

We use TF-IDF Vectorization with:

max_features=50,000

ngram_range=(1, 2) → unigrams + bigrams

English stopword removal

This converts text into numerical vectors suitable for ML models.

🤖 Machine Learning Models Used

We trained and evaluated three models:

1️⃣ Logistic Regression

High performance on text data

Fast and lightweight

Best overall accuracy

2️⃣ SVM (LinearSVC)

Excellent for high-dimensional spaces

Very close performance to Logistic Regression

3️⃣ KNN

Poor performance for text

Slow on high-dimensional vectors

Included for comparison purposes

📊 Results
Model Accuracy
Model	Accuracy
Logistic Regression	89.96%
SVM (LinearSVC)	89.85%
KNN	78.24%
Conclusion

Logistic Regression performs best and is selected as the final model for sentiment classification.

📉 Confusion Matrix Interpretation

TP → Correctly predicted positive

TN → Correctly predicted negative

FP → Incorrectly predicted positive

FN → Incorrectly predicted negative

Logistic Regression & SVM show high TP and TN, indicating strong predictive performance.

🧪 How to Run the Project
1. Clone the repository
git clone <your-repo-url>
cd movie_review_sentiment_analysis

2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run notebooks

Open Jupyter Notebook or VS Code and run:

notebooks/01_EDA.ipynb

notebooks/02_DataCleaning.ipynb

notebooks/03_ML_Models.ipynb

🧠 Project Structure
movie_review_sentiment_analysis/
│
├── dataset/
│   ├── raw/
│   │   └── IMDB_Dataset.csv
│   └── cleaned/
│       └── cleaned_reviews.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_DataCleaning.ipynb
│   └── 03_ML_Models.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── utils.py
│
├── results/
│   ├── accuracy_plot.png
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_svm.png
│   └── confusion_matrix_knn.png
│
├── README.md
└── requirements.txt

🌐 Future Improvements

✨ Add a Streamlit web app for live predictions
✨ Use deep learning models (LSTM, BERT)
✨ Deploy model with AWS or HuggingFace
✨ Add explainability (LIME / SHAP)