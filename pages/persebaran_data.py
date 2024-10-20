import streamlit as st
import pandas as pd
import altair as alt
import plost


st.set_page_config(layout='wide')
x_df = pd.read_csv('./dataset/dataset_predicted_sentiment.csv')
top_user_df = pd.read_csv('./dataset/top_users_data.csv')

with open('./css/style.css') as f :
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def card_metric(label, value):
    return f"""
    <div class="css-1r6slb0 e1tzin5v2">
        <h6>{label}</h6>
        <p style='font-size:32px; font-weight:600'>{value}</p>
    </div>
    """

def donut_chart(selected_theta, data1):
    plost.donut_chart(
        data=data1,
        theta=selected_theta,
        color='company',
        legend='bottom',
        use_container_width=True
    )

def main():
    st.title('Analisis Sentiment data tweet')
    label_df = x_df['predict_sentiment'].value_counts().reset_index()# Mengganti nama kolom
    st.write(label_df)
    # data = {
    #     'company': ['Positif', 'Negatif', 'Netral'],
    #     'jumlah kata': [label_df.iloc[0]['count'], label_df.iloc[1]['count'], label_df.iloc[2]['count']],
    # }
    #data = pd.read_csv('./dataset/label_count.csv')
    #data = pd.DataFrame(data)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(card_metric('Jumlah kata positif', label_df.iloc[0]['count']), unsafe_allow_html=True)
    with col2:
        st.markdown(card_metric('Jumlah kata negatif', label_df.iloc[1]['count']), unsafe_allow_html=True)
    with col3:
        st.markdown(card_metric('Jumlah kata netral', label_df.iloc[2]['count']), unsafe_allow_html=True)
    col1, col2 = st.columns(3)
    donut_theta = 'jumlah kata'
    donut_chart(donut_theta, data)

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
