📌 FUTURE_ML_03 — Resume Ranking System

Machine Learning Internship — Task 03
This repository contains a powerful Resume Ranking System built with Streamlit and Machine Learning (NLP) to help you upload, process, score, rank, and visualize candidate resumes based on their relevance to a job description.

🧠 Overview

This project implements an intelligent system that:

🔹 Loads and preprocesses resume data (CSV format)
🔹 Cleans and analyzes text using NLTK and Scikit-Learn
🔹 Computes similarity between resumes and a job description using TF-IDF
🔹 Ranks candidates using a customizable scoring formula
🔹 Displays insights and visual analytics via an interactive Streamlit web app

✨ Built with Python, this system is great for HR screening, candidate matching, and small-scale recruitment workflows.

🚀 Features
✅ Upload & Preprocess

✔ Upload resume CSV with a “Resume” column
✔ Handles text cleaning & lemmatization
✔ Removes stopwords & punctuation

📊 Intelligent Ranking

Ranking is based on three weighted components:

Component	Purpose
Similarity Score	Matches resumes to job description using TF-IDF
Experience Matcher	Detects years of experience from text
Length Modifier	Rewards more detailed resumes

Weights can be adjusted live via sliders.

📈 Interactive Visual Analytics

The dashboard automatically generates:

✔ Score distribution charts
✔ Word count and experience histograms
✔ Similarity vs Experience scatter plot
✔ Ranked resume table with preview

📦 How It Works

Load Required Libraries
Python libraries such as pandas, nltk, sklearn, seaborn, matplotlib, and streamlit are used.

Text Preprocessing
Uses NLTK for stopwords & lemmatization
Cleans text of punctuation, digits, and noise

TF-IDF Model
Computes similarity between resumes and a given job description
Scores computed via cosine similarity

Scoring & Ranking
Final score = Weighted combination of similarity, experience, and resume length

Visualization
Built-in charts to help compare candidate scores visually

🛠️ Installation & Run

Clone this project and install dependencies:

git clone https://github.com/nehasrikaruku-netizen/FUTURE_ML_03.git
cd FUTURE_ML_03
pip install -r requirements.txt

Start the Streamlit app:

streamlit run streamlit_app.py

Upload your resume CSV and start ranking candidates!

📂 Repo Structure
FUTURE_ML_03/
├── LICENSE
├── README.md
├── future_ml_03.ipynb        # Notebook implementation
├── resume.csv                # Sample resume data
├── streamlit_app.py          # Main Streamlit application
📜 Example Usage

Once you upload your CSV:

✔ Choose weights for similarity, experience, and length
✔ Enter job description text
✔ Click Calculate Similarity Scores
✔ View ranked candidates + analytics

🧰 Libraries & Tools Used

✔ Python
✔ Streamlit – Interactive UI
✔ NLTK – Preprocessing
✔ Scikit-Learn – TF-IDF & similarity
✔ Pandas – Data handling
✔ Matplotlib & Seaborn – Visualization

📌 Enhancements You Can Make

🔹 Add resume file support (PDF / DOCX)
🔹 Add custom NLP models (BERT, spaCy, etc.)
🔹 Deploy on Cloud (Heroku, AWS, Streamlit Cloud)
🔹 Export top candidates to PDF/Excel

🧾 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

❤️ Acknowledgements

This project was developed as part of a Machine Learning Internship task, combining practical NLP and Streamlit visualization for end-to-end candidate ranking.
