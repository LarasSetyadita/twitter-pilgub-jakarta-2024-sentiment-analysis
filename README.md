# twitter-pilgub-jakarta-2024-sentiment-analysis

## Deskripsi Proyek 
Proyek ini bertujuan untuk melakukan <b>Analisis Sentiment</b> pada data Twitter yang berhubungan dengan keyword <b>"jakarta
menyala"</b> dalam konteks <b>Pemilihan Gubernur Jakarta 2024</b>. Dengan menggunakan teknik <i>machine learning</i>, model 
ini mengkategorikan tweet menjadi sentiment positif, negatif, atau netral. Proyek ini memanfaatkan aplikasi Streamlit sebagai
antarmuka untuk analisis dan visualisasi. 

## Fitur Utama
- <b>Preprocessing Teks :</b> Menggunakan stemming dengan Sastrawi dan penghilangan kata-kata yang tidak relevan (stopwords) dari bahasa Indonesia.
- <b>Model Naive Bayes :</b> Untuk klasifikasi sentimen menggunakan bag of words. 
- <b>Visualisasi :</b> Menampilkan hasil analisis dalam bentuk kartu mentrik dan <i>donut chart</i> untuk distribusi sentiment.
- <b>Form Analisis Interaktif :</b> Pengguna dapat memasukkan teks untuk dianalisis secara real-time.

## Instalasi
1. Clone repository ini : 
<br><code>git clone https://github.com/username/repository-name.git</code>
2. Install dependencies :
<br><code> pip instal -r requirements.txt</code></br>
3. Download NLTK stopwords : 
<br><code><br>import nltk</br>
nltk.download('stopwords')</code></br>

## Menjalankan Aplikasi
Gunakan perintah berikut untuk menjalankan aplikasi Streamlit!
<code>streamlit run app.py</code>

## Teknologi yang digunakan
1. <b>Python</b> (NLTK, Pandas, Sastrawi)
2. <b>Streamlit</b> - Untuk antarmuka web.
3. <b>Pickle</b> - Untuk memuat model dan fitur.
4. <b>CSS</b> - Untuk styling tampilan aplikasi.
