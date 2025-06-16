import streamlit as st
# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ratnakar\Downloads\cleaned_audible_catalog.csv")
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join([word for word in text.split() if word not in stop_words])

    df['Cleaned_Description'] = df['Description'].apply(clean_text)
    return df

df = load_data()

# TF-IDF & Cosine Similarity (once only)
@st.cache_resource
def prepare_similarity(df):
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['Cleaned_Description'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return tfidf_matrix, sim_matrix

tfidf_matrix, sim_matrix = prepare_similarity(df)

# Content-Based Recommender
def recommend_content(title, top_n=5):
    if title not in df['Book Name'].values:
        return ["Book not found."]
    idx = df[df['Book Name'] == title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i[0]]['Book Name'] for i in sim_scores]

# Cluster-Based Recommender
def recommend_cluster(title, top_n=5):
    if title not in df['Book Name'].values:
        return ["Book not found."]
    cluster_id = df[df['Book Name'] == title]['Cluster'].values[0]
    cluster_books = df[(df['Cluster'] == cluster_id) & (df['Book Name'] != title)]
    top_books = cluster_books.sort_values(by='Rating', ascending=False).head(top_n)
    return top_books['Book Name'].tolist()

# Streamlit UI
st.title("📚 Book Recommendation App")
st.markdown("Get personalized book recommendations based on content or clusters.")

book_list = df['Book Name'].sort_values().unique().tolist()
selected_book = st.selectbox("Choose a book you liked:", book_list)

model_choice = st.radio("Select Recommendation Method:", ["Content-Based", "Cluster-Based"])

if st.button("Get Recommendations"):
    if model_choice == "Content-Based":
        recs = recommend_content(selected_book)
    else:
        recs = recommend_cluster(selected_book)

    st.subheader("📖 Recommended Books:")
    for book in recs:
        st.markdown(f"- {book}")

# Optional EDA

with st.expander("🎯 Top Authors by Avg Rating"):
    top_authors = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_authors)