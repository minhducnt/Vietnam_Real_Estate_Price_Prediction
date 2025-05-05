from transformers import AutoTokenizer, RobertaForSequenceClassification
import torch
from tqdm import tqd
from underthesea import sentiment
import pandas as pd
import pyhdfs
from pyspark.sql import *
from pyspark.sql import functions as func

# Khởi tạo mô hình và tokenizer
checkpoint = "wonrax/phobert-base-vietnamese-sentiment"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
model = RobertaForSequenceClassification.from_pretrained(checkpoint)

def get_sentiment(text, batch_size=32):
    # Định nghĩa mapping từ nhãn model sang nhãn mong muốn
    sentiment_mapping = {
        "POS": "Tích cực",
        "NEU": "Trung tính",
        "NEG": "Tiêu cực"
    }

    num_batches = len(text) // batch_size + (1 if len(text) % batch_size != 0 else 0)
    dominant_sentiment = ''

    for i in tqd(range(num_batches), desc="Processing Batches"):
        batch_texts = text[i*batch_size:(i+1)*batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

        # Get predictions and find dominant sentiment
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for prediction in predictions:
            scores = prediction.tolist()
            highest_score_index = scores.index(max(scores))
            label_mapping = model.config.id2label  # Sửa chỗ này để lấy mapping từ id sang label
            dominant_sentiment_label = label_mapping[highest_score_index]

            # Map nhãn model sang nhãn mong muốn
            dominant_sentiment = sentiment_mapping[dominant_sentiment_label]

    return dominant_sentiment

output_file = "comments_sentiment.csv"

# Khởi tạo SparkSession
spark = SparkSession.builder.getOrCreate()

# Đọc dữ liệu từ file CSV và sử dụng dòng đầu tiên làm tên cột
df_spark = spark.read.csv("hdfs://localhost:9000/csvspark/preprocessed_comments.csv", header=True)

# Chuyển đổi DataFrame của Spark thành DataFrame của pandas
df = df_spark.toPandas()

df['sentiment']= df['content'].apply(get_sentiment)

# Lưu dataframe vào file csv
df.to_csv(output_file, index=False)

# Khởi tạo client kết nối đến Hadoop
client = pyhdfs.HdfsClient(hosts='127.0.0.1:9870', user_name='dr.who')
print(client)

# Copy file csv từ local vào HDFS
hdfs_path = '/csvspark/comments_sentiment.csv'
client.copy_from_local(output_file, hdfs_path)

print(f"Uploaded {output_file} to {hdfs_path} on HDFS")

# Trích xuất dữ liệu đánh giá mẫu từ bộ dữ liệu để tiến hành so sánh và lựa chọn mô hình
comments = [
    "Chưa xài nên chưa biết viết cảm nhận sao",
    "Sản phẩm dùng tốt lắm, lần sau sẽ ủng hộ",
    "Giao thiếu quà tặng hũ 30ml",
    "Tôi sử dụng sp dòng lão hóa này được 10 năm tôi rất hài lòng.",
    "mua của shop mấy sản phẩm mà hơi thất vọng vì sử dụng không được như ý. dù đã nhắn tin riêng để phản hồi với shop nhưng kh thấy shop phản hồi gì cả",
    "Đã nhận sản phẩm. Đầy đủ quà tặng. Đã mua sản phẩm lần thú 3. Nhưng đây là hãng lớn mà giao sản phẩm không có hộp đựng gì cả thiếu chuyên nghiệp",
    "hàng tặng thì vơi có 1/2 chai. Tặng thì tặng đàng hoàng",
    "Hàng nhận hơi muộn so với khi đặt. Kem mỏng, thẩm thấu nhanh. Quả tặng bạt ngàn, và màu son rất đẹp.",
    "Khá thất vọng vì sản phẩm như kiểu hàng tồn kho, cũ và bẩn",
    "Sao k dc tặng quà như trên hình nhỉ",
    "đợi sử dụng có kết quả rồi quay lại sao"
]

labels = []

# Phân tích cảm xúc bằng underthesea
for comment in comments:
    label = sentiment(comment)
    labels.append(label)

print(labels)

# Phân tích cảm xúc bằng phoBert
for comment in comments:
    label = get_sentiment(comment)
    labels.append(label)

print(labels)