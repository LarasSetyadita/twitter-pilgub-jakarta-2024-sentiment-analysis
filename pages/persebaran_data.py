import streamlit as st
import pandas as pd

x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')

def main():
    st.title('Analisis Sentiment data tweet')
    st.table(x_df)
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.subheader('Jumlah Tweet Positif per User')
    #     #st.bar_chart(x_df.set_index('username')['positive'], use_container_width=True)
    # with col2:
    #     st.write('iya')
    # with col3:
    #     st.write('iya')

if __name__ == '__main__':
     main()
