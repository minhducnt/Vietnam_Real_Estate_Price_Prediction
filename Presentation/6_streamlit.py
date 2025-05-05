import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyspark.sql import *
from pyspark.sql import functions as func

# Khởi tạo SparkSession
spark = SparkSession.builder.getOrCreate()

# Đọc dữ liệu từ file CSV và sử dụng dòng đầu tiên làm tên cột
df_spark = spark.read.csv("hdfs://localhost:9000/csvspark/comments_sentiment.csv", header=True)

# Chuyển đổi DataFrame của Spark thành DataFrame của pandas
df = df_spark.toPandas()

# Tạo danh sách các sản phẩm
product_titles = df['title'].unique().tolist()

# Tạo dropdown để chọn sản phẩm
selected_title = st.selectbox('Chọn sản phẩm', product_titles)

# Lọc các loại của sản phẩm theo title đã chọn
variants = df[df['title'] == selected_title]['variant'].unique().tolist()

# Tạo dropdown để chọn loại của sản phẩm
selected_variant = st.selectbox('Chọn loại của sản phẩm', variants)

# Tạo dropdown để chọn loại đánh giá
evaluate_options = df['sentiment'].unique().tolist()
selected_evaluate = st.selectbox('Chọn loại đánh giá', evaluate_options)

# Lọc các nội dung đánh giá theo các lựa chọn đã chọn
filtered_reviews = df[(df['title'] == selected_title) & 
                      (df['variant'] == selected_variant) & 
                      (df['sentiment'] == selected_evaluate)]['content']

selected = df[(df['title'] == selected_title) & 
                      (df['variant'] == selected_variant) & 
                      (df['sentiment'] == selected_evaluate)]

st.write(f"Brand: {selected['brand'].iloc[0]}")
st.write(f"Giá: {selected['price'].iloc[0]}")

# Hiển thị các nội dung đánh giá
st.write('Các nội dung đánh giá:')
for review in filtered_reviews:
    st.write('- ' + review)

all_reviews_text = ' '.join(filtered_reviews)

# Tạo WordCloud từ chuỗi đánh giá
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews_text)

# Hiển thị WordCloud bằng Streamlit
st.write('WordCloud của các từ khóa xuất hiện nhiều nhất:')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)