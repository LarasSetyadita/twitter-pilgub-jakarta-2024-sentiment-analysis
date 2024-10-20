import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud



x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')

def buat_word_cloud(teks, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    st.pyplot(plt)

def main():
    st.markdown('Analisis Word Cloud')
    option = st.selectbox('Pilih Word Cloud yang ingin Anda Lihat', ('Positif', 'Negatif', 'Netral'))
    if option == 'Positif':
        df_positif = x_df[x_df['predict_sentiment'] == 'positive']
        teks_positif = " ".join(tweet for tweet in df_positif['tweet'].fillna(''))
        buat_word_cloud(teks_positif, "Word Cloud - Tweet Positif")
    elif option == 'Negatif':
        df_negatif = x_df[x_df['predict_sentiment'] == 'negative']
        teks_negatif = " ".join(tweet for tweet in df_negatif['tweet'].fillna(''))
        buat_word_cloud(teks_negatif, "Word Cloud - Tweet Negatif")
    else:
        df_netral = x_df[x_df['predict_sentiment'] == 'neutral']
        teks_netral = " ".join(tweet for tweet in df_netral['tweet'].fillna(''))
        buat_word_cloud(teks_netral, "Word Cloud - Tweet Netral")

if __name__ == '__main__':
     main()