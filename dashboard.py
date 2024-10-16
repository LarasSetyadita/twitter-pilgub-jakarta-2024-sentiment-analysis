import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import joblib, os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data and models
data_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

nltk.download('stopwords')

# Initialize vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(data_df['tweet'])

def preprocess(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    stop_words.update(['user', 'http'])
    
    text = text.lower()
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = re.sub(r'[^a-zA-Z\s]', '', t)
        if t not in stop_words and t != '@user':
            t = stemmer.stem(t)
            new_text.append(t)
    return " ".join(new_text)

def perform_kmeans(df):
    return df

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores

def predict_sentiment(text):
    processed_text = preprocess(text)
    print(f"Processed Text: {processed_text}")  # Check output after preprocessing
    
    text_vector = vectorizer.transform([processed_text])  # Ensure it's 2D
    print(f"Text Vector Shape: {text_vector.shape}")  # Check shape
    
    prediction = model.predict(text_vector)
    return prediction[0]

def analyze_words(text):
    words = text.split()
    sentiments = {word: analyze_sentiment(word) for word in words}
    return sentiments

def dashboard_page():
    st.image('23115.jpg')
    st.title("Analisis Sentimen Pemilihan Gubernur Jakarta 2024")
    st.subheader('Data Twitter')
    with st.form('myform'):
        st.subheader("Let's Try")
        text_input = st.text_area("Ketikkan kata atau kalimat", placeholder="Masukkan teks di sini")
        submit_button = st.form_submit_button("Analisa")

        # Summary of sentiment

    if submit_button:
        if text_input:
            processed_text = preprocess(text_input)
            prediction = predict_sentiment(processed_text)
            print(f'Text: {text_input}')
            print(f'Sentiment: {text_input}')
            word_sentiment = analyze_words(text_input)
            
            overall_sentiment = {
                'positif': sum(score['pos'] for score in word_sentiment.values()),
                'negatif': sum(score['neg'] for score in word_sentiment.values()),
                'netral': sum(score['neu'] for score in word_sentiment.values())
            }

            highest_sentiment = max(overall_sentiment, key=overall_sentiment.get)
            if highest_sentiment == 'positif':
                st.info('Positif')
            elif highest_sentiment == 'negatif':
                st.warning('Negatif')
            else:
                st.info('Netral')


            words = list(word_sentiment.keys())
            positif_scores = [score['pos'] for score in word_sentiment.values()]
            negatif_scores = [score['neg'] for score in word_sentiment.values()]
            netral_scores = [score['neu'] for score in word_sentiment.values()]
            data = {
                'Words': words,
                'Positive': positif_scores,
                'Negative': negatif_scores,
                'Neutral': netral_scores,
            }
            word_df = pd.DataFrame(data)
            x = np.arange(len(words))  # Label for X axis
            st.subheader("Tingkat Sentimen Keseluruhan")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.pie(overall_sentiment.values(), labels=('Positif', 'Negatif', 'Netral'),
                       autopct='%1.1f%%', startangle=90,
                       colors=['#66c2a5', '#fc8d62', '#8da0cb'])
                ax.axis('equal')
                st.pyplot(fig)
            with col2:
                st.bar_chart(word_df.set_index('Words'))
            
            
        else:
            st.subheader('tingkat sentiment perkata')
            st.warning("Silakan masukkan teks untuk dianalisis.")

def word_cloud_page():
    st.subheader('Word Cloud')
    option = st.selectbox('Pilih Word Cloud yang ingin Anda Lihat', ('Positif', 'Negatif', 'Netral'))
    if option == 'Positif':
        df_positif = data_df[data_df['predict_sentiment'] == 'positive']
        teks_positif = " ".join(tweet for tweet in df_positif['tweet'])
        buat_word_cloud(teks_positif, "Word Cloud - Tweet Positif")
    elif option == 'Negatif':
        df_negatif = data_df[data_df['predict_sentiment'] == 'negative']
        teks_negatif = " ".join(tweet for tweet in df_negatif['tweet'])
        buat_word_cloud(teks_negatif, "Word Cloud - Tweet Negatif")
    else:
        df_netral = data_df[data_df['predict_sentiment'] == 'neutral']
        teks_netral = " ".join(tweet for tweet in df_netral['tweet'])
        buat_word_cloud(teks_netral, "Word Cloud - Tweet Netral")

def buat_word_cloud(teks, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    st.pyplot(plt)

# Sidebar
st.sidebar.title("Navigasi")

# State and logic
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

if st.sidebar.button("Dashboard"):
    st.session_state.page = 'Dashboard'
if st.sidebar.button("Word Cloud"):
    st.session_state.page = 'Word Cloud'

if st.session_state.page == 'Dashboard':
    dashboard_page()
elif st.session_state.page == 'Word Cloud':
    word_cloud_page()
