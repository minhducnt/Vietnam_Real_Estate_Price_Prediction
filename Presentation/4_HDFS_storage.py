import pyhdfs

output_file = 'preprocessed_comments.csv'  # Đường dẫn tới file CSV

# Khởi tạo client kết nối đến Hadoop
client = pyhdfs.HdfsClient(hosts='127.0.0.1:9870', user_name='dr.who')
print(client)

# Copy file csv từ local vào HDFS
hdfs_path = '/csvspark/preprocessed_comments.csv'
client.copy_from_local(output_file, hdfs_path)

print(f"Uploaded {output_file} to {hdfs_path} on HDFS")