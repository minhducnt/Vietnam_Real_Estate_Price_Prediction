import os
import subprocess
from pyspark.sql import SparkSession

def initialize_spark_session(app_name="HDFSStorage"):
    """Khởi tạo và trả về một phiên Spark."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

def upload_to_hdfs(local_path, hdfs_path):
    """Đưa một file từ hệ thống file cục bộ lên HDFS."""
    try:
        # Tạo thư mục HDFS nếu nó chưa tồn tại
        subprocess.run(["hadoop", "fs", "-mkdir", "-p", os.path.dirname(hdfs_path)], check=True)

        # Đưa file lên HDFS
        subprocess.run(["hadoop", "fs", "-put", "-f", local_path, hdfs_path], check=True)

        print(f"Đã tải lên thành công {local_path} vào HDFS tại {hdfs_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tải tập tin lên HDFS: {e}")
        return False

def read_from_hdfs(spark, hdfs_path):
    """Đọc file CSV từ HDFS vào DataFrame Spark."""
    try:
        df = spark.read.option("header", True).csv(hdfs_path)
        print(f"Đã đọc thành công {df.count()} bản ghi từ {hdfs_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc tập tin từ HDFS: {e}")
        return None

def write_to_hdfs(df, hdfs_path, format="csv"):
    """Ghi DataFrame Spark vào HDFS."""
    try:
        if format == "csv":
            df.coalesce(1).write.option("header", True).mode("overwrite").csv(hdfs_path)
        elif format == "parquet":
            df.write.mode("overwrite").parquet(hdfs_path)
        print(f"Đã ghi thành công DataFrame vào HDFS tại {hdfs_path}")
        return True
    except Exception as e:
        print(f"Lỗi khi ghi DataFrame vào HDFS: {e}")
        return False

def list_hdfs_files(hdfs_dir):
    """Liệt kê các file trong một thư mục HDFS."""
    try:
        result = subprocess.run(["hadoop", "fs", "-ls", hdfs_dir],
                               capture_output=True, text=True, check=True)
        print(f"Các tập tin trong {hdfs_dir}:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi liệt kê thư mục HDFS: {e}")
        return None

def delete_from_hdfs(hdfs_path):
    """Xóa một file hoặc thư mục từ HDFS."""
    try:
        subprocess.run(["hadoop", "fs", "-rm", "-r", hdfs_path], check=True)
        print(f"Đã xóa thành công {hdfs_path} từ HDFS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi xóa tập tin từ HDFS: {e}")
        return False

if __name__ == "__main__":
    # Ví dụ sử dụng
    spark = initialize_spark_session()

    # Đường dẫn cục bộ và HDFS
    local_file = "processed_data/part-00000-*.csv"  # Path to the processed CSV file
    hdfs_dir = "hdfs://localhost:9000/vietnam_real_estate"
    hdfs_file = f"{hdfs_dir}/processed_real_estate_data.csv"

    # Upload the file to HDFS
    upload_to_hdfs(local_file, hdfs_file)

    # Liệt kê các file trong thư mục HDFS
    list_hdfs_files(hdfs_dir)

    # Đọc file từ HDFS
    df = read_from_hdfs(spark, hdfs_file)

    if df is not None:
        # Hiển thị dữ liệu mẫu
        print("Dữ liệu mẫu từ HDFS:")
        df.show(5)

        # Ghi lại vào HDFS dưới định dạng Parquet (hiệu quả hơn cho Spark)
        write_to_hdfs(df, f"{hdfs_dir}/processed_real_estate_data.parquet", format="parquet")

    # Dừng phiên Spark
    spark.stop()
