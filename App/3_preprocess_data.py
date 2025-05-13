from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, split, trim, element_at, lit, sum, mean
from pyspark.sql.types import DoubleType, IntegerType, FloatType
import re
import os

def initialize_spark_session(app_name="RealEstatePreprocessing"):
    """Khởi tạo và trả về một phiên Spark."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """Đọc dữ liệu từ file CSV vào DataFrame Spark."""
    df = spark.read.option("header", True).option("encoding", "utf-8").csv(file_path)
    print(f"Loaded {df.count()} records from {file_path}")
    print("Original schema:")
    df.printSchema()
    return df

def check_missing_values(df):
    """Kiểm tra và in số lượng giá trị bị thiếu trong mỗi cột."""
    print("Missing values per column:")
    df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
    return df

def drop_unnecessary_columns(df, columns_to_drop):
    """Loại bỏ các cột không cần thiết khỏi DataFrame."""
    df_cleaned = df.drop(*columns_to_drop)
    print(f"Dropped columns: {columns_to_drop}")
    return df_cleaned

def handle_null_values(df):
    """Xử lý các giá trị null trong DataFrame."""
    # Xử lý các cột số - thay thế null bằng -1
    numeric_columns = ['floor_num', 'toilet_num', 'livingroom_num', 'bedroom_num']
    for column in numeric_columns:
        df = df.withColumn(
            column,
            when(col(column).isNull(), -1).otherwise(col(column))
        )

    # Xử lý các cột phân loại - thay thế null bằng "Không xác định"
    categorical_columns = ['category', 'direction', 'liability', 'location']
    for column in categorical_columns:
        df = df.withColumn(
            column,
            when(col(column).isNull(), 'Không xác định').otherwise(col(column))
        )

    print("Handled null values")
    return df

def process_location(df):
    """Xử lý cột vị trí để trích xuất quận/huyện và thành phố/tỉnh."""
    # Trích xuất quận/huyện và thành phố/tỉnh từ vị trí
    df = df.withColumn(
        'district',
        trim(element_at(split(regexp_replace(col('location'), r'\s*Lưu tin\s*$', ''), ','), -2))
    ).withColumn(
        'city_province',
        trim(element_at(split(regexp_replace(col('location'), r'\s*Lưu tin\s*$', ''), ','), -1))
    )

    # Loại bỏ cột vị trí gốc
    df = df.drop('location')

    print("Processed location column into district and city_province")
    return df

def process_price(df):
    """Xử lý cột giá để chuyển sang định dạng số."""
    from pyspark.sql.functions import expr, regexp_replace

    # Xử lý cột giá
    df = df.withColumn('price',
        when(
            col("price").like("%tỷ%"),
            expr("cast(regexp_replace(regexp_replace(price, 'tỷ', ''), ',', '.') as double) * 1000000000")
        ).when(
            col("price").like("%triệu%"),
            expr("cast(regexp_replace(regexp_replace(price, 'triệu', ''), ',', '.') as double) * 1000000")
        ).when(
            col("price").like("%Thương lượng%"),
            lit(None).cast("double")
        ).otherwise(
            regexp_replace(col("price"), ",", "").cast("double")
        ).cast("float")
    )

    # Loại bỏ các dòng có giá trị null
    df = df.filter(col("price").isNotNull())

    print("Processed price column")
    return df

def process_area(df):
    """Xử lý cột diện tích để chuyển sang định dạng số."""
    # Xử lý cột diện tích - loại bỏ 'm2' và chuyển sang float
    df = df.withColumn('area',
        regexp_replace(col('area'), r'\s*m2$', '').cast('float')
    )

    # Loại bỏ các dòng có diện tích null hoặc bằng 0
    df = df.filter((col("area").isNotNull()) & (col("area") > 0))

    print("Processed area column")
    return df

def process_floor_num(df):
    """Xử lý cột số tầng để chuyển sang định dạng số."""
    # Loại bỏ 'tầng' và chuyển sang số nguyên
    df = df.withColumn('floor_num',
        regexp_replace(col('floor_num').cast('string'), r'\s*tầng$', '').cast('int')
    )

    print("Processed floor_num column")
    return df

def process_bedroom_num(df):
    """Xử lý cột số phòng ngủ để chuyển sang định dạng số."""
    # Loại bỏ 'phòng' và chuyển sang số nguyên
    df = df.withColumn('bedroom_num',
        regexp_replace(col('bedroom_num').cast('string'), r'\s*phòng$', '').cast('int')
    )

    print("Processed bedroom_num column")
    return df

def process_street(df):
    """Xử lý cột đường để chuyển sang định dạng số."""
    # Xử lý độ rộng đường
    df = df.withColumn("street_clean", regexp_replace(col("street"), "m", ""))

    # Handle ranges like "a-b"
    df = df.withColumn("split", split(col("street_clean"), "-"))

    # Calculate average for ranges or use the single value
    df = df.withColumn(
        "street_width",
        when(
            col("street_clean").isNull(), lit(-1)
        ).when(
            (col("split").getItem(1).isNotNull()),
            (col("split").getItem(0).cast("float") + col("split").getItem(1).cast("float")) / 2
        ).otherwise(
            col("split").getItem(0).cast("float")
        )
    )

    # Replace null values with -1
    df = df.withColumn(
        "street_width",
        when(col("street_width").isNull(), lit(-1)).otherwise(col("street_width"))
    )

    # Drop temporary columns
    df = df.drop("street", "street_clean", "split")

    print("Processed street column into street_width")
    return df

def calculate_price_per_m2(df):
    """Calculate price per square meter."""
    df = df.withColumn(
        "price_per_m2",
        when(col("area") > 0, col("price") / col("area")).otherwise(None)
    )

    # Filter out unreasonable price_per_m2 values
    df = df.filter(
        (col("price_per_m2") >= 2e6) & (col("price_per_m2") <= 1e8)
    )

    print("Calculated price_per_m2")
    return df

def convert_data_types(df):
    """Convert columns to appropriate data types."""
    df = df.withColumn("area", col("area").cast("float")) \
           .withColumn("floor_num", col("floor_num").cast("int")) \
           .withColumn("toilet_num", col("toilet_num").cast("int")) \
           .withColumn("livingroom_num", col("livingroom_num").cast("int")) \
           .withColumn("bedroom_num", col("bedroom_num").cast("int")) \
           .withColumn("street_width", col("street_width").cast("float")) \
           .withColumn("price", col("price").cast("float")) \
           .withColumn("price_per_m2", col("price_per_m2").cast("float"))

    print("Converted data types")
    return df

def rename_columns(df):
    """Rename columns for clarity."""
    df = df.withColumnRenamed("area", "area_m2") \
           .withColumnRenamed("street_width", "street_width_m")

    print("Renamed columns")
    return df

def save_processed_data(df, output_path):
    """Save the processed DataFrame to CSV."""
    # Save as a single CSV file
    df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

    # Find the CSV file in the output directory
    csv_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
    if csv_files:
        csv_file = csv_files[0]
        print(f"Data saved to {output_path}/{csv_file}")
    else:
        print(f"Data saved to {output_path}")

    return output_path

def preprocess_data(input_file, output_path):
    """Main function to preprocess real estate data."""
    # Initialize Spark session
    spark = initialize_spark_session()

    try:
        # Load data
        df = load_data(spark, input_file)

        # Check missing values
        df = check_missing_values(df)

        # Data cleaning steps
        df = drop_unnecessary_columns(df, ['link', 'title'])
        df = handle_null_values(df)
        df = process_location(df)
        df = process_price(df)
        df = process_area(df)
        df = process_floor_num(df)
        df = process_bedroom_num(df)
        df = process_street(df)
        df = calculate_price_per_m2(df)
        df = convert_data_types(df)
        df = rename_columns(df)

        # Final check of missing values
        df = check_missing_values(df)

        # Save processed data
        output_path = save_processed_data(df, output_path)

        print(f"Preprocessing completed. Final record count: {df.count()}")

        return output_path

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    # File paths
    input_file = "raw_real_estate_data.csv"
    output_dir = "processed_data"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run preprocessing
    preprocess_data(input_file, output_dir)
