import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
import os

def initialize_spark_session(app_name="RealEstateVisualization"):
    """Khởi tạo và trả về một phiên Spark."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

def load_processed_data(spark, data_path):
    """Đọc dữ liệu đã xử lý từ CSV."""
    df = spark.read.option("header", True).csv(data_path)
    # Convert to Pandas for visualization
    pandas_df = df.toPandas()
    print(f"Đã tải {len(pandas_df)} bản ghi cho việc trực quan hóa")
    return pandas_df

def create_output_dir(output_dir="visualizations"):
    """Tạo thư mục để lưu trữ các hình ảnh trực quan hóa."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_price_distribution(df, output_dir):
    """Vẽ biểu đồ phân phối giá."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Biểu đồ phân phối giá
    sns.histplot(df["price_per_m2"], kde=True, ax=ax[0])
    ax[0].set_title("Phân Phối Giá")
    ax[0].set_xlabel("Giá trên m\u00b2 (VND)")
    ax[0].set_ylabel("Số lượng")

    # Biểu đồ phân phối giá đã chuyển đổi logarit
    sns.histplot(np.log1p(df["price_per_m2"]), kde=True, ax=ax[1])
    ax[1].set_title("Phân Phối Giá Đã Chuyển Đổi Logarit")
    ax[1].set_xlabel("Log(Giá trên m\u00b2)")
    ax[1].set_ylabel("Số lượng")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_distribution.png", dpi=300)
    plt.close()

    # Biểu đồ boxplot theo thành phố
    plt.figure(figsize=(14, 8))
    city_data = df.groupby("city_province")["price_per_m2"].median().sort_values(ascending=False).head(10).reset_index()

    sns.boxplot(x="city_province", y="price_per_m2", data=df[df["city_province"].isin(city_data["city_province"])])
    plt.title("Phân Phối Giá Theo 10 Tỉnh/Thành Phố Hàng Đầu")
    plt.xlabel("Tỉnh/Thành Phố")
    plt.ylabel("Giá trên m\u00b2 (VND)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_by_city_boxplot.png", dpi=300)
    plt.close()

    print("Đã tạo các biểu đồ phân phối giá")

def plot_feature_relationships(df, output_dir):
    """Vẽ biểu đồ mối quan hệ giữa các đặc trưng chính và giá."""
    # Biểu đồ Diện tích - Giá
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="area_m2", y="price_per_m2", data=df, alpha=0.5)
    plt.title("Mối Quan Hệ Giữa Diện Tích và Giá")
    plt.xlabel("Diện tích (m\u00b2)")
    plt.ylabel("Giá trên m\u00b2 (VND)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/area_vs_price.png", dpi=300)
    plt.close()

    # Ma trận tương quan
    numeric_columns = ["area_m2", "floor_num", "toilet_num", "livingroom_num", "bedroom_num", "street_width_m", "price_per_m2"]
    correlation = df[numeric_columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Ma Trận Tương Quan Giữa Các Đặc Trưng Số")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300)
    plt.close()

    # Biểu đồ Đặc trưng - Giá
    for feature in ["bedroom_num", "floor_num", "toilet_num"]:
        plt.figure(figsize=(10, 6))
        feature_data = df.groupby(feature)["price_per_m2"].mean().reset_index()
        sns.barplot(x=feature, y="price_per_m2", data=feature_data)
        plt.title(f"Average Price by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Average Price per m² (VND)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/price_by_{feature}.png", dpi=300)
        plt.close()

    print("Đã tạo các biểu đồ mối quan hệ giữa các đặc trưng")

def plot_categorical_comparisons(df, output_dir):
    """Vẽ biểu đồ so sánh giá theo các đặc trưng phân loại."""
    categorical_features = ["category", "direction", "liability"]

    for feature in categorical_features:
        # Tính giá trung bình theo giá trị đặc trưng
        feature_data = df.groupby(feature)["price_per_m2"].mean().sort_values(ascending=False).reset_index()

        # Vẽ biểu đồ cột
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature, y="price_per_m2", data=feature_data)
        plt.title(f"Average Price by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Average Price per m² (VND)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/price_by_{feature}.png", dpi=300)
        plt.close()

    print("Đã tạo các biểu đồ so sánh theo phân loại")

def plot_time_trends(df, output_dir):
    """Vẽ biểu đồ xu hướng theo thời gian nếu có dữ liệu thời gian."""
    # Tạo một trường thời gian giả nếu không có
    # Trong tình huống thực tế, bạn có thể có một dấu thời gian hoặc ngày đăng
    if "posting_date" in df.columns:
        # Trích xuất các đặc trưng thời gian và vẽ biểu đồ xu hướng
        df["posting_month"] = pd.to_datetime(df["posting_date"]).dt.month

        # Vẽ biểu đồ xu hướng giá theo tháng
        monthly_data = df.groupby("posting_month")["price_per_m2"].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x="posting_month", y="price_per_m2", data=monthly_data, marker="o")
        plt.title("Average Price Trend by Month")
        plt.xlabel("Month")
        plt.ylabel("Average Price per m² (VND)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/price_trend_by_month.png", dpi=300)
        plt.close()

        print("Time trend plots created")

def create_interactive_map(df, output_dir):
    """Tạo bản đồ tương tác các vị trí bất động sản."""
    # Đây là phiên bản đơn giản hóa không có mã hóa địa lý thực tế
    # Trong ứng dụng thực tế, bạn cần mã hóa địa chỉ để lấy vị độ/kinh độ

    # For demonstration, we'll create a simple map of Vietnam
    # and place some markers based on city/province

    # Define approximate coordinates for major cities in Vietnam
    city_coords = {
        "Hà Nội": [21.0285, 105.8542],
        "Hồ Chí Minh": [10.8231, 106.6297],
        "Đà Nẵng": [16.0544, 108.2022],
        "Hải Phòng": [20.8449, 106.6881],
        "Cần Thơ": [10.0452, 105.7469],
        "Nha Trang": [12.2388, 109.1967],
        "Huế": [16.4637, 107.5909],
        "Vũng Tàu": [10.3461, 107.0842]
    }

    # Create a map centered on Vietnam
    vietnam_map = folium.Map(location=[16.0544, 108.2022], zoom_start=5)

    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(vietnam_map)

    # Add a marker for each major city with price info
    for city, coords in city_coords.items():
        city_data = df[df["city_province"].str.contains(city, case=False, na=False)]

        if len(city_data) > 0:
            avg_price = city_data["price_per_m2"].mean()
            avg_price_text = f"{avg_price:,.0f} VND/m\u00b2"

            folium.Marker(
                location=coords,
                popup=f"<strong>{city}</strong><br>Giá Trung Bình: {avg_price_text}<br>Số BĐS: {len(city_data)}",
                tooltip=f"{city}: {avg_price_text}",
                icon=folium.Icon(color="blue", icon="home")
            ).add_to(marker_cluster)

    # Save the map as HTML
    vietnam_map.save(f"{output_dir}/property_map.html")

    print("Đã tạo bản đồ tương tác")

def create_plotly_visualizations(df, output_dir):
    """Tạo các biểu đồ tương tác bằng Plotly."""
    # Phân phối giá theo thành phố
    city_prices = df.groupby("city_province")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
    city_prices = city_prices.head(10)

    fig = px.bar(
        city_prices,
        x="city_province",
        y="price_per_m2",
        title="Top 10 Cities by Average Price per m²",
        labels={"city_province": "City/Province", "price_per_m2": "Average Price per m² (VND)"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(f"{output_dir}/top_cities_price.html")

    # Price vs. Area scatter plot
    fig = px.scatter(
        df.sample(min(1000, len(df))),
        x="area_m2",
        y="price_per_m2",
        color="city_province",
        size="bedroom_num",
        hover_data=["district", "bedroom_num", "floor_num"],
        title="Price vs. Area by City/Province",
        labels={
            "area_m2": "Area (m²)",
            "price_per_m2": "Price per m² (VND)",
            "city_province": "City/Province",
            "bedroom_num": "Bedrooms"
        }
    )
    fig.update_layout(height=600)
    fig.write_html(f"{output_dir}/price_vs_area_interactive.html")

    # 3D visualization
    fig = px.scatter_3d(
        df.sample(min(1000, len(df))),
        x="area_m2",
        y="bedroom_num",
        z="price_per_m2",
        color="city_province",
        size="floor_num",
        hover_data=["district", "category"],
        title="3D Visualization of Property Features",
        labels={
            "area_m2": "Area (m²)",
            "bedroom_num": "Bedrooms",
            "price_per_m2": "Price per m² (VND)",
            "city_province": "City/Province",
            "floor_num": "Floors"
        }
    )
    fig.update_layout(height=700)
    fig.write_html(f"{output_dir}/3d_visualization.html")

    print("Interactive Plotly visualizations created")

def create_dashboard_template(output_dir):
    """Create a simple HTML dashboard template."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vietnam Real Estate Market Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <header class="bg-blue-600 text-white p-4 rounded-lg shadow-lg mb-8">
                <h1 class="text-3xl font-bold">Vietnam Real Estate Market Analysis</h1>
                <p class="mt-2">Interactive Dashboard for Property Price Analysis</p>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price Distribution</h2>
                    <img src="price_distribution.png" alt="Price Distribution" class="w-full">
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by City</h2>
                    <img src="price_by_city_boxplot.png" alt="Price by City" class="w-full">
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by Bedrooms</h2>
                    <img src="price_by_bedroom_num.png" alt="Price by Bedrooms" class="w-full">
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by Floors</h2>
                    <img src="price_by_floor_num.png" alt="Price by Floors" class="w-full">
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by Toilets</h2>
                    <img src="price_by_toilet_num.png" alt="Price by Toilets" class="w-full">
                </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow mb-8">
                <h2 class="text-xl font-semibold mb-4">Feature Correlations</h2>
                <img src="correlation_matrix.png" alt="Correlation Matrix" class="w-full">
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by Property Type</h2>
                    <img src="price_by_category.png" alt="Price by Category" class="w-full">
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price by Direction</h2>
                    <img src="price_by_direction.png" alt="Price by Direction" class="w-full">
                </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow mb-8">
                <h2 class="text-xl font-semibold mb-4">Interactive Property Map</h2>
                <iframe src="property_map.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Price vs. Area</h2>
                    <iframe src="price_vs_area_interactive.html" width="100%" height="400px" frameborder="0"></iframe>
                </div>
                <div class="bg-white p-4 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Top Cities by Price</h2>
                    <iframe src="top_cities_price.html" width="100%" height="400px" frameborder="0"></iframe>
                </div>
            </div>

            <div class="bg-white p-4 rounded-lg shadow mb-8">
                <h2 class="text-xl font-semibold mb-4">3D Feature Visualization</h2>
                <iframe src="3d_visualization.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>

            <footer class="bg-blue-600 text-white p-4 rounded-lg text-center">
                <p>© 2025 Vietnam Real Estate Price Prediction Project</p>
            </footer>
        </div>
    </body>
    </html>
    """

    with open(f"{output_dir}/dashboard.html", "w") as f:
        f.write(html_content)

    print("Dashboard template created")

def create_visualizations(data_path, output_dir="visualizations"):
    """Main function to create all visualizations."""
    # Initialize Spark
    spark = initialize_spark_session()

    try:
        # Create output directory
        output_dir = create_output_dir(output_dir)

        # Load data
        df = load_processed_data(spark, data_path)

        # Create static visualizations
        plot_price_distribution(df, output_dir)
        plot_feature_relationships(df, output_dir)
        plot_categorical_comparisons(df, output_dir)
        plot_time_trends(df, output_dir)

        # Create interactive visualizations
        create_interactive_map(df, output_dir)
        create_plotly_visualizations(df, output_dir)

        # Create dashboard template
        create_dashboard_template(output_dir)

        print(f"All visualizations created in {output_dir}")
        return output_dir

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    # Path to processed data
    data_path = "processed_data/part-00000-*.csv"  # Adjust based on your file naming

    # Create visualizations
    create_visualizations(data_path)
