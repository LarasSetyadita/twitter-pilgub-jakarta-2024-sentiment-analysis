from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import streamlit as st
import plost
import pandas as pd
import pickle
import re
import nltk


#####################
## import css file ##
#####################

with open('./css/style.css') as f :
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


#################
## Import Data ##
#################
x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')
data = {
    'company': ['Positif', 'Negatif', 'Netral'],
    'jumlah kata': [100, 150, 200],
    'polaritas': [0.1, -0.2, 0.3]
}
data = pd.DataFrame(data)

###################
## Import models ##
###################

with open('./models/feature-bow.p', 'rb') as f:
    feature_bow = pickle.load(f)
with open('./models/model-nb.p', 'rb') as f:
    model_nb = pickle.load(f)


#feature_bow.fit(x_df['tweet'])
####################
## feature method ##
####################

# chart models
def altair_chart(selected_theta, data1):
    st.write('dalam proses pembuatan')

def donut_chart(selected_theta, data1):
    plost.donut_chart(
        data=data1,
        theta=selected_theta,
        color='company',
        legend='bottom',
        use_container_width=True
    )
# method for preprocess text
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
        t = re.sub(r'[^a-zA-Z@\s]', '', t)  # Menghapus karakter non-alfabet
        if t not in stop_words and t != "":  # Pastikan kata tidak kosong
            # Melakukan stemming
            t = stemmer.stem(t)
            new_text.append(t)
    return " ".join(new_text)

def create_sentiment_dataframe(processed_text):
    model_naive_bayes = [0, 0, 0]

    text_array = processed_text.split()

    # Prediksi untuk model Naive Bayes
    polarity_model1 = model_nb.predict(feature_bow.transform(text_array))
    for polarity in polarity_model1:
        if polarity == 'Positive':
            model_naive_bayes[0] += 1
        elif polarity == 'Negative':
            model_naive_bayes[1] += 1
        else:
            model_naive_bayes[2] += 1

    data = {
        'Polarity': ['Positive', 'Negative', 'Neutral'],
        'Model Naive Bayes': model_naive_bayes,
    }

    df = pd.DataFrame(data)
    return df

#################
## main method ##
#################

def main():

    # header page
    st.image('./images/pilkada-header-logo.jpg')
    st.title('Analisis Sentiment Pemilihan Gubernur Jakarta 2024')
    st.subheader('Crawling data didapatkan dari aplikasi X')
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # deskripsi
    st.header("Deskripsi Sentiment Analysis")
    st.write('Analisis sentimen ini bertujuan untuk memahami opini publik '
                  'terkait keyword "jakarta menyala" yang berkaitan dengan '
                  'pemilihan gubernur Jakarta 2024. Data dikumpulkan melalui '
                  'crawling dari Twitter, di mana keyword tersebut mungkin '
                  'mencerminkan aspirasi, kritik, atau harapan masyarakat '
                  'terhadap calon gubernur Jakarta. Analisis ini dilakukan '
                  'dengan menggunakan teknik machine learning untuk '
                  'mengkategorikan tweet menjadi sentimen positif, '
                  'negatif, atau netral.')
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # form
    st.header('Ayo mulai menganalisis')
    with st.form('form_sentiment'):
        text_input = st.text_area("Let's Try", placeholder="Ketikkan kata atau kalimat")
        submit_button = st.form_submit_button('Analisis')


    if submit_button:
        if text_input:
            if text_input:
                processed_text = preprocess(text_input)
                st.write("Processed Text:", processed_text)
                featured_text = feature_bow.transform([processed_text])
                sentiment_nb = model_nb.predict(featured_text)[0]
                if sentiment_nb == 'Positive':
                    st.info("Prediksi Naive Bayes: Positif")
                elif sentiment_nb == 'Negative':
                    st.error("Prediksi Naive Bayes: Negatif")
                else:  # Neutral case
                    st.warning("Prediksi Naive Bayes: Neutral")

                # # Create sentiment DataFrame
                word_df = create_sentiment_dataframe(processed_text)  # Call function for dataframe
                # st.sidebar.subheader('Donut chart parameter')


                # Check for empty DataFrame before plotting
                # if not word_df.empty:
                #     donut_chart(donut_theta, word_df)
                # else:
                #     st.warning("Data untuk donut chart kosong.")

                # donut_theta = st.sidebar.selectbox('Select data', ('Model Naive Bayes', 'Model Neural Network'))

                def card_metric(label, value):
                    return f"""
                    <div class="css-1r6slb0 e1tzin5v2">
                        <h6>{label}</h6>
                        <p style='font-size:32px; font-weight:600'>{value}</p>
                    </div>
                    """

                # Membuat tampilan metrik dalam bentuk grid dengan kolom
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(card_metric('Jumlah kata positif', word_df.iloc[0, 1]), unsafe_allow_html=True)
                with col2:
                    st.markdown(card_metric('Jumlah kata negatif', word_df.iloc[1, 1]), unsafe_allow_html=True)
                with col3:
                    st.markdown(card_metric('Jumlah kata netral', word_df.iloc[2, 1]), unsafe_allow_html=True)

if __name__ == '__main__':
     main()



