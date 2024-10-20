import streamlit as st
import pandas as pd
import altair as alt
st.set_page_config(layout='wide')
x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')
top_user_df = pd.read_csv('./dataset/top_users_data.csv')

def main():
    st.title('Analisis Sentiment data tweet')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**User dengan tweet positif terbanyak**')
        positive_count = top_user_df.groupby('username')['positive'].sum().reset_index()  # Hitung jumlah positif per username
        chart = alt.Chart(positive_count).mark_bar().encode(
            x=alt.X('username', sort=alt.SortField('positive', order='descending')),  # Menjaga urutan dari DataFrame
            y='positive:Q'
        ).properties(
            width=400,
            height=300
        ).configure()

        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown('**User dengan tweet negative terbanyak**')
        negative_count = top_user_df.groupby('username')[
            'negative'].sum().reset_index()  # Hitung jumlah positif per username
        chart = alt.Chart(negative_count).mark_bar().encode(
            x=alt.X('username', sort=alt.SortField('negative', order='descending')),  # Menjaga urutan dari DataFrame
            y='negative:Q'
        ).properties(
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)
    with col3:
        st.markdown('**User dengan tweet netral terbanyak**')
        neutral_count = top_user_df.groupby('username')[
            'neutral'].sum().reset_index()  # Hitung jumlah positif per username
        chart = alt.Chart(neutral_count).mark_bar().encode(
            x=alt.X('username', sort=alt.SortField('neutral', order='descending')),  # Menjaga urutan dari DataFrame
            y='neutral:Q'
        ).properties(
            width=400,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
     main()
