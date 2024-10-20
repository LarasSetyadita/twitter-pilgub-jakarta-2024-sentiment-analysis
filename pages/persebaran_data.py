import streamlit as st
import pandas as pd

x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')

def main():
    st.markdown('ini adalah halaman wordcloud')
    my_table = st.dataframe(x_df)

if __name__ == '__main__':
     main()
