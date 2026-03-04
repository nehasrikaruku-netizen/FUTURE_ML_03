import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# DOWNLOAD REQUIRED NLTK DATA
# ==========================================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_data()

# ==========================================
# INITIALIZE GLOBALS
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==========================================
# TEXT PREPROCESSING
# ==========================================
def preprocess_text(text):
    """Clean and preprocess resume text"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    
    return " ".join(words)

def simple_clean(text):
    """Simple cleaning for job descriptions"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def extract_experience(text):
    """Extract years of experience from resume text"""
    if pd.isna(text):
        return 0
    
    text = str(text).lower()
    pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    matches = re.findall(pattern, text)
    
    if matches:
        years = [int(match) for match in matches]
        return max(years)
    return 0

# ==========================================
# STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Resume Ranking System", layout="wide")

st.title("🎯 Resume Ranking System")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Weight configuration
    st.subheader("Scoring Weights")
    weight_similarity = st.slider(
        "Similarity Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        help="Weight for job description match"
    )
    weight_experience = st.slider(
        "Experience Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Weight for years of experience"
    )
    weight_length = st.slider(
        "Resume Length Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="Weight for resume completeness"
    )
    
    # Normalize weights
    total_weight = weight_similarity + weight_experience + weight_length
    if total_weight > 0:
        weight_similarity /= total_weight
        weight_experience /= total_weight
        weight_length /= total_weight
    
    st.info(f"✓ Weights normalized to 1.0")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Resume Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with resumes",
        type=['csv'],
        help="CSV should have 'Resume' column with resume text"
    )

with col2:
    st.subheader("📋 Quick Info")
    if uploaded_file:
        resume_df = pd.read_csv(uploaded_file)
        st.metric("Loaded Resumes", len(resume_df))
    else:
        st.metric("Loaded Resumes", 0)

# Process uploaded file
if uploaded_file:
    resume_df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(resume_df)} resumes")
else:
    st.info("📌 Upload a CSV file to get started")
    st.stop()

# ==========================================
# DISPLAY DATASET INFO
# ==========================================
with st.expander("📊 Dataset Overview", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resumes", len(resume_df))
    with col2:
        st.metric("Columns", len(resume_df.columns))
    with col3:
        st.metric("Missing Values", resume_df.isnull().sum().sum())
    
    st.dataframe(resume_df.head(5), use_container_width=True)

# ==========================================
# DETECT RESUME COLUMN
# ==========================================
possible_columns = ['Resume_str', 'Resume', 'resume', 'text', 'resume_text', 'Text']
resume_column = None

for col in possible_columns:
    if col in resume_df.columns:
        resume_column = col
        break

if resume_column is None:
    st.error("❌ No resume text column found. Expected: 'Resume', 'resume', or 'Resume_str'")
    st.write("Available columns:", list(resume_df.columns))
    st.stop()

st.success(f"✅ Using '{resume_column}' column for resume text")

# ==========================================
# PREPROCESSING
# ==========================================
with st.spinner("🔄 Preprocessing resumes..."):
    resume_df['cleaned_resume'] = resume_df[resume_column].apply(preprocess_text)
    resume_df['word_count'] = resume_df[resume_column].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    resume_df['experience_years'] = resume_df[resume_column].apply(extract_experience)

st.success("✅ Preprocessing completed")

# ==========================================
# JOB DESCRIPTION INPUT
# ==========================================
st.subheader("💼 Job Description")

job_description = st.text_area(
    "Enter the job description to match against resumes:",
    value="""Looking for a Python developer with strong experience in machine learning,
natural language processing, data analysis, and model deployment.""",
    height=120,
    help="Resumes will be scored based on similarity to this job description"
)

if st.button("🔍 Calculate Similarity Scores", use_container_width=True):
    with st.spinner("⏳ Computing TF-IDF and similarity scores..."):
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=7000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(resume_df['cleaned_resume'])
        
        # Clean and transform job description
        job_cleaned = simple_clean(job_description)
        job_vector = vectorizer.transform([job_cleaned])
        
        # Compute similarity
        similarity_scores = cosine_similarity(tfidf_matrix, job_vector).flatten()
        resume_df['similarity_score'] = similarity_scores
    
    st.success("✅ Similarity scores calculated")
else:
    # Use default similarity if not calculated
    if 'similarity_score' not in resume_df.columns:
        resume_df['similarity_score'] = 0.0

# ==========================================
# SCORING & RANKING
# ==========================================
st.subheader("📈 Candidate Scoring")

# Normalize features
max_exp = resume_df['experience_years'].max()
max_wc = resume_df['word_count'].max()

resume_df['exp_norm'] = resume_df['experience_years'] / (max_exp + 1) if max_exp > 0 else 0
resume_df['length_norm'] = resume_df['word_count'] / (max_wc + 1) if max_wc > 0 else 0

# Weighted final score
resume_df['final_score'] = (
    weight_similarity * resume_df['similarity_score'] +
    weight_experience * resume_df['exp_norm'] +
    weight_length * resume_df['length_norm']
)

# Sort by final score
ranked_resumes = resume_df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

# Display top candidates
num_top_candidates = st.slider(
    "Number of top candidates to display",
    min_value=5,
    max_value=min(50, len(ranked_resumes)),
    value=min(10, len(ranked_resumes))
)

st.dataframe(
    ranked_resumes[['similarity_score', 'experience_years', 'word_count', 'final_score']].head(num_top_candidates),
    use_container_width=True
)

# ==========================================
# VISUALIZATIONS
# ==========================================
st.subheader("📊 Analytics & Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Experience", "Word Count", "Score Breakdown"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resume_df['final_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title("Final Score Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Final Score")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resume_df['experience_years'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax.set_title("Years of Experience Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tab3:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(resume_df['word_count'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_title("Resume Word Count Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    plt.close()

with tab4:
    # Scatter plot of key metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        resume_df['experience_years'],
        resume_df['similarity_score'],
        s=resume_df['word_count']/2,
        c=resume_df['final_score'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Similarity Score")
    ax.set_title("Resume Metrics Correlation (size = word count, color = final score)", fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Final Score")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ==========================================
# DETAILED CANDIDATE VIEW
# ==========================================
st.subheader("👤 Detailed Candidate View")

if len(ranked_resumes) > 0:
    candidate_idx = st.selectbox(
        "Select a candidate to view details:",
        options=range(len(ranked_resumes)),
        format_func=lambda i: f"Rank #{i+1} (Score: {ranked_resumes.iloc[i]['final_score']:.3f})"
    )
    
    selected_candidate = ranked_resumes.iloc[candidate_idx]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Score", f"{selected_candidate['final_score']:.3f}")
    with col2:
        st.metric("Similarity", f"{selected_candidate['similarity_score']:.3f}")
    with col3:
        st.metric("Experience", f"{int(selected_candidate['experience_years'])} years")
    with col4:
        st.metric("Word Count", f"{int(selected_candidate['word_count'])} words")
    
    st.subheader("Resume Content")
    resume_text = str(selected_candidate[resume_column])
    display_text = resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text
    st.text_area(
        "Raw Resume:",
        value=display_text,
        height=200,
        disabled=True
    )

# ==========================================
# EXPORT RESULTS
# ==========================================
st.subheader("💾 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Export CSV
    csv_data = ranked_resumes[['similarity_score', 'experience_years', 'word_count', 'final_score']].to_csv(index=False)
    st.download_button(
        label="📥 Download Ranked Results (CSV)",
        data=csv_data,
        file_name="ranked_candidates.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    # Export Top 10 with full resumes
    top_10_export = ranked_resumes.head(10)[[resume_column, 'similarity_score', 'experience_years', 'final_score']].copy()
    csv_full = top_10_export.to_csv(index=False)
    st.download_button(
        label="📥 Download Top 10 (with resumes)",
        data=csv_full,
        file_name="top_10_candidates.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown("📝 Resume Ranking System | Built with Streamlit | ML-Powered Candidate Matching")
