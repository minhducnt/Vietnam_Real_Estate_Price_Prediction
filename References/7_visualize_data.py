import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from pyspark.sql import *
from pyspark.sql import functions as func

# Khởi tạo SparkSession
spark = SparkSession.builder.getOrCreate()

# Đọc dữ liệu từ file CSV và sử dụng dòng đầu tiên làm tên cột
df_spark = spark.read.csv("hdfs://localhost:9000/csvspark/preprocessed_comments.csv", header=True)

# Chuyển đổi DataFrame của Spark thành DataFrame của pandas
data = df_spark.toPandas()

# Đếm số lượng đánh giá theo số sao cho từng brand
star_counts = data.groupby(['brand', 'stars']).size().unstack(fill_value=0)

# Tính tổng số lượng đánh giá cho mỗi thương hiệu
star_counts['total'] = star_counts.sum(axis=1)

# Sắp xếp thương hiệu theo số lượng đánh giá từ cao đến thấp và chọn 10 thương hiệu đầu tiên
top_brands = star_counts.sort_values(by='total', ascending=False).head(10).index

# Lọc dữ liệu cho các thương hiệu hàng đầu
star_counts_top = star_counts.loc[top_brands]

# Chuẩn bị dữ liệu cho boxplot
star_counts_top = star_counts_top.reset_index()
star_counts_top = star_counts_top.melt(id_vars='brand', var_name='stars', value_name='count')

# Tạo boxplot
fig, ax = plt.subplots(figsize=(10, 6))
boxplot_data = [group['count'].values for name, group in star_counts_top.groupby('brand')]
ax.boxplot(boxplot_data, labels=star_counts_top['brand'].unique())
ax.set_xlabel('Thương hiệu')
ax.set_ylabel('Số sao')
ax.set_title('Biểu đồ phân phối số sao đánh giá theo top 10 thương hiệu')

# Hiển thị boxplot trên Streamlit
st.pyplot(fig)




top_products = data['title'].value_counts().nlargest(10).index

top_df = data[data['title'].isin(top_products)]

star_distribution = top_df.groupby(['title', 'stars']).size().unstack(fill_value=0)

# Vẽ biểu đồ cột
fig, ax = plt.subplots(figsize=(15, 10))

# Duyệt qua từng sản phẩm và vẽ phân phối số sao
for product in star_distribution.index:
    ax.bar(star_distribution.columns, star_distribution.loc[product], label=product)

# Thiết lập nhãn và tiêu đề
ax.set_xlabel('Stars')
ax.set_ylabel('Number of Reviews')
ax.set_title('Biểu đồ phân phối số sao đánh giá theo top 10 sản phẩm')
ax.legend(title='Products')

# Hiển thị biểu đồ
plt.xticks(range(6))

# Sử dụng Streamlit để hiển thị biểu đồ
st.pyplot(fig)


# def load_data():
#     data = pd.read_csv('preprocessed_comments.csv')
#     return data

# data = load_data()

# # Chia giá thành 7 khoảng giá
# data['price_bins'] = pd.cut(data['price'], bins=7, labels=[f'Khoảng {i+1}' for i in range(7)])

# # Vẽ biểu đồ scatter plot
# fig, ax = plt.subplots()
# scatter = ax.scatter(data['price_bins'], data['stars'], alpha=0.6, edgecolors='w', linewidth=0.5)

# # Thêm tiêu đề và nhãn
# ax.set_title('Mối quan hệ giữa giá sản phẩm và số sao')
# ax.set_xlabel('Khoảng giá')
# ax.set_ylabel('Số sao')

# # Hiển thị biểu đồ trên Streamlit
# st.pyplot(fig)

# Đọc dữ liệu từ file CSV
# data = pd.read_csv("preprocessed_comments.csv")

# Tính toán doanh thu cho từng thương hiệu

# Chuyển đổi kiểu dữ liệu nếu cần
data['quantity_sold'] = data['quantity_sold'].astype(float)
data['price'] = data['price'].astype(float)

# Tiếp tục tính toán
sales_by_brand = data.groupby("brand").apply(lambda x: (x["quantity_sold"] * x["price"]).sum())
print(sales_by_brand)
# Lấy danh sách các thương hiệu và doanh số tương ứng
brands = sales_by_brand.index.tolist()
sales = sales_by_brand.values.tolist()

# Tạo biểu đồ
fig, ax = plt.subplots()

# Thiết lập thông tin cho biểu đồ
bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] # Màu cho mỗi thanh cột
ax.bar(brands[:10], sales[:10], color=bar_colors[:10]) # Chỉ lấy top 10 thương hiệu

ax.set_ylabel('Doanh số bán hàng')
ax.set_title('Doanh số bán hàng theo thương hiệu (Top 10)')
ax.set_xticklabels(brands[:10], rotation=45, ha='right') # Chỉ hiển thị top 10 thương hiệu
ax.set_xlabel('Thương hiệu')

st.pyplot(fig)

# def load_data_5(file_path):
#     data = pd.read_csv(file_path)
#     return data

# # Lọc dữ liệu duy nhất theo tiêu đề sản phẩm và đếm số lượt đánh giá theo số sao
# def process_data(data):
#     unique_titles = data['title'].unique()
#     aggregated_data = []

#     for title in unique_titles:
#         title_data = data[data['title'] == title]
#         star_counts = title_data['stars'].value_counts().to_dict()
#         for star in range(6):
#             if star not in star_counts:
#                 star_counts[star] = 0
#         total_quantity_sold = title_data['quantity_sold'].sum()
#         aggregated_data.append({'title': title, 'stars': star_counts, 'quantity_sold': total_quantity_sold})

#     return pd.DataFrame(aggregated_data)

# # Vẽ biểu đồ Bubble chart
# def draw_bubble_chart(data):
#     fig, ax = plt.subplots()
#     for index, row in data.iterrows():
#         for star, count in row['stars'].items():
#             if count > 0:
#                 ax.scatter(star, row['quantity_sold'], s=count*10, alpha=0.5)

#     ax.set_xlabel('Stars')
#     ax.set_ylabel('Quantity Sold')
#     ax.set_title('Sales vs. Stars')
#     ax.legend()
#     st.pyplot(fig)

# st.title('Sales vs. Stars')
# data = load_data_5("preprocessed_comments.csv")
# processed_data = process_data(data)
# draw_bubble_chart(processed_data)
# def load_data_6(file_path):
#     return pd.read_csv(file_path)

# # Load data
# file_path = "preprocessed_comments.csv"  # Thay đổi đường dẫn tới file CSV của bạn
# data_6 = load_data_6(file_path)

# # Loại bỏ các dòng trùng lặp theo tiêu đề sản phẩm
# unique_titles = data_6.drop_duplicates(subset="title", keep="first")

# # Tạo biểu đồ
# def create_sales_by_price_chart(data_6):
#     # Tạo histogram hoặc bar chart
#     plt.figure(figsize=(10, 6))
#     plt.bar(data_6["price"], data_6["quantity_sold"], color='skyblue', edgecolor='black', width=1.2)
#     plt.xlabel('Price')
#     plt.ylabel('Quantity Sold')
#     plt.title('Sales by Price Range')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     st.pyplot()

# # Hiển thị biểu đồ trên Streamlit
# st.title('Sales by Price Range')
# create_sales_by_price_chart(unique_titles)

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv("preprocessed_comments.csv")

# # Loại bỏ các dòng trùng lặp dựa trên cột "title"
# unique_titles = data.drop_duplicates(subset="title")

# # Tạo biểu đồ
# @st.cache
# def create_sales_chart(data):
#     price_ranges = pd.cut(data['price'], bins=10)  # Chia dữ liệu thành 10 khoảng giá
#     sales_by_price = data['quantity_sold'] # Tính tổng doanh số bán hàng trong từng khoảng giá
#     print(sales_by_price)
#     plt.bar(sales_by_price.index.astype(str), sales_by_price.values)
#     plt.xlabel('Price Range')
#     plt.ylabel('Quantity Sold')
#     plt.title('Sales by Price Range')

# # Streamlit app
# st.title('Sales by Price Range')

# # Hiển thị biểu đồ
# create_sales_chart(unique_titles)
# st.pyplot()
# data_6 = pd.read_csv("preprocessed_comments.csv")

# # Lọc dữ liệu duy nhất theo title
# unique_data = data_6.drop_duplicates(subset="title")

# # Tạo biểu đồ
# def create_sales_chart(data_6):
#     plt.figure(figsize=(10, 6))
#     plt.bar(data_6['price'], data_6['quantity_sold'], color='blue')
#     plt.xlabel('Price')
#     plt.ylabel('Quantity Sold')
#     plt.title('Sales by Price Range')
#     plt.xticks(rotation=45)
#     st.pyplot()
# st.title("Sales Analysis")

# # Hiển thị dữ liệu
# st.write("Sample data:")
# st.write(unique_data)

#     # Vẽ biểu đồ
# create_sales_chart(unique_data)

# def draw_bar_chart(data):
#     # Nhóm dữ liệu theo giá và tính tổng quantity_sold trong mỗi nhóm
#     grouped_data = data.groupby('price')['quantity_sold'].sum().reset_index()

#     # Vẽ biểu đồ
#     plt.figure(figsize=(10, 6))
#     plt.bar(grouped_data['price'], grouped_data['quantity_sold'], color='skyblue')
#     plt.xlabel('Price')
#     plt.ylabel('Total Quantity Sold')
#     plt.title('Total Quantity Sold at Different Price Ranges')
#     plt.xticks(rotation=45)
#     st.pyplot()
# draw_bar_chart(unique_data)


# # Read CSV into pandas
# data = pd.read_csv("preprocessed_comments.csv")
# unique_data = data.drop_duplicates(subset="title")

# unique_data.head()
# df = pd.DataFrame(unique_data)
 
# name = df['price'].head(12)
# price = df['quantity_sold'].head(12)
 
# # Figure Size
# fig, ax = plt.subplots(figsize =(10, 7))
# ax.bar(name[0:10], price[0:10])
 
# # Customize plot
# plt.xlabel('Price')
# plt.ylabel('Quantity Sold')
# plt.title('Top 10 Products by Quantity Sold')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# plt.show()
# # Display the chart using Streamlit
# st.pyplot(fig)

