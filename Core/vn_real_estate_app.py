import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import os
import pickle
import random
import time
import subprocess
from pyngrok import ngrok

# Khởi tạo biến toàn cục để lưu tên cột
FEATURE_COLUMNS = {
    'area': 'area (m2)',
    'street': 'street (m)'
}

# Thiết lập trang với giao diện hiện đại
st.set_page_config(
    page_title="Dự Đoán Giá Bất Động Sản Việt Nam",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh để tạo giao diện hiện đại
st.markdown('''
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Roboto', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c9aff;
        color: white;
    }
    .stButton>button {
        background-color: #4c9aff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #3d7ecc;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
''', unsafe_allow_html=True)

# Khởi tạo phiên Spark
@st.cache_resource
def get_spark_session():
    """Khởi tạo và trả về một phiên Spark."""
    return (
        SparkSession.builder
        .appName("VNRealEstatePricePrediction")
        .config("spark.driver.memory", "2g")
        .master("local[*]")
        .getOrCreate()
    )

# Đọc dữ liệu
@st.cache_data
def load_data(file_path="../Data/Final Data Cleaned.csv"):
    """Đọc dữ liệu bất động sản từ file CSV."""
    try:
        # Đọc dữ liệu bằng pandas
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {e}")
        return pd.DataFrame()

@st.cache_data
def preprocess_data(data):
    """Tiền xử lý dữ liệu cho phân tích và mô hình hóa."""
    # Tạo bản sao để tránh cảnh báo của Pandas
    df = data.copy()

    # Đổi tên cột để dễ sử dụng (nếu chưa có)
    column_mapping = {
        'area (m2)': 'area_m2',
        'street (m)': 'street_width_m'
    }

    # Kiểm tra và đổi tên cột nếu cần
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Xử lý giá trị thiếu
    numeric_cols = ["area_m2", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street_width_m"]
    for col in numeric_cols:
        if col in df:
            # Thay thế -1 (giá trị thiếu) bằng giá trị trung vị
            median_val = df[df[col] != -1][col].median()
            df[col] = df[col].replace(-1, median_val)

    # Chuyển đổi logarithm cho giá
    df['price_log'] = np.log1p(df['price_per_m2'])

    return df

# Chuyển đổi dữ liệu pandas sang spark
@st.cache_resource
def convert_to_spark(data):
    """Chuyển đổi DataFrame pandas sang DataFrame Spark."""
    spark = get_spark_session()
    return spark.createDataFrame(data)

# Huấn luyện mô hình
@st.cache_resource
def train_model(data):
    """Huấn luyện mô hình dự đoán giá bất động sản."""
    # Lưu trữ tên cột gốc để sử dụng cho dự đoán sau này
    global FEATURE_COLUMNS

    # Kiểm tra các cột có sẵn trong dữ liệu
    data_columns = data.columns

    # Xác định tên cột đúng cho các đặc trưng số
    area_column = 'area_m2' if 'area_m2' in data_columns else 'area (m2)'
    street_column = 'street_width_m' if 'street_width_m' in data_columns else 'street (m)'

    # Lưu cấu trúc cột để sử dụng cho dự đoán
    FEATURE_COLUMNS = {
        'area': area_column,
        'street': street_column
    }

    # Chuyển đổi dữ liệu sang Spark DataFrame
    spark_df = convert_to_spark(data)

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

    # Chuẩn bị các đặc trưng, sử dụng tên cột thực tế
    numeric_features = [area_column, "bedroom_num", "floor_num", "toilet_num", "livingroom_num", street_column]

    # Tạo vector đặc trưng
    assembler = VectorAssembler(
        inputCols=numeric_features,
        outputCol="features"
    )

    # Chuẩn hóa đặc trưng
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )

    # Mô hình GBT Regressor
    gbt = GBTRegressor(
        featuresCol="scaled_features",
        labelCol="price_log",
        maxDepth=5,
        maxIter=100
    )

    # Tạo pipeline
    pipeline = Pipeline(stages=[assembler, scaler, gbt])

    # Huấn luyện mô hình
    model = pipeline.fit(train_df)

    # Đánh giá mô hình
    predictions = model.transform(test_df)

    evaluator = RegressionEvaluator(
        labelCol="price_log",
        predictionCol="prediction",
        metricName="r2"
    )

    r2 = evaluator.evaluate(predictions)

    evaluator.setMetricName("rmse")
    rmse = evaluator.evaluate(predictions)

    return model, r2, rmse

# Hàm dự đoán giá
def predict_price(model, input_data):
    """Dự đoán giá dựa trên đầu vào của người dùng."""
    try:
        global FEATURE_COLUMNS

        # Tạo bản sao của dữ liệu đầu vào
        data_copy = input_data.copy()

        # Điều chỉnh tên cột để phù hợp với mô hình
        if hasattr(FEATURE_COLUMNS, 'get'):
            # Đảm bảo area_m2 và street_width_m được đổi tên phù hợp
            if 'area_m2' in data_copy and FEATURE_COLUMNS.get('area') != 'area_m2':
                data_copy[FEATURE_COLUMNS['area']] = data_copy['area_m2']
                del data_copy['area_m2']

            if 'street_width_m' in data_copy and FEATURE_COLUMNS.get('street') != 'street_width_m':
                data_copy[FEATURE_COLUMNS['street']] = data_copy['street_width_m']
                del data_copy['street_width_m']
        else:
            # Nếu FEATURE_COLUMNS chưa được khởi tạo, sử dụng giá trị mặc định
            # Áp dụng mapping cũ cho trường hợp này
            column_mapping = {
                'area_m2': 'area (m2)',
                'street_width_m': 'street (m)'
            }
            for new_name, old_name in column_mapping.items():
                if new_name in data_copy and old_name not in data_copy:
                    data_copy[old_name] = data_copy[new_name]

        spark = get_spark_session()

        # In ra dữ liệu đầu vào để gỡ lỗi
        print("Dữ liệu đầu vào dự đoán:", data_copy)

        # Tạo DataFrame từ đầu vào
        input_df = spark.createDataFrame([data_copy])

        # Thực hiện dự đoán
        result = model.transform(input_df)

        # Lấy kết quả
        prediction_log = result.select("prediction").collect()[0][0]

        # Chuyển từ giá trị logarithm sang giá trị thật
        predicted_price = np.expm1(prediction_log)

        return predicted_price
    except Exception as e:
        # Ghi lại lỗi để gỡ rối
        print(f"Lỗi khi dự đoán: {e}")
        st.error(f"Có lỗi xảy ra khi dự đoán: {e}")
        return 0

# Tạo hàm để chạy ngrok
def run_ngrok():
    """Kết nối ứng dụng Streamlit với ngrok để tạo URL public."""
    # Thiết lập ngrok - Người dùng cần nhập authtoken
    st.sidebar.subheader("Kết nối Ngrok")

    ngrok_auth_token = st.sidebar.text_input("Nhập Ngrok Authtoken", type="password")

    if ngrok_auth_token:
        try:
            # Thiết lập authtoken
            ngrok.set_auth_token(ngrok_auth_token)

            # Tạo tunnel HTTP đến cổng 8501 (cổng mặc định của Streamlit)
            public_url = ngrok.connect(addr="8501", proto="http").public_url

            st.sidebar.success("✅ Ngrok đã kết nối thành công!")
            st.sidebar.markdown(f"**URL public:** {public_url}")
            st.sidebar.markdown("Chia sẻ URL này để người khác có thể truy cập ứng dụng của bạn.")

            # Lưu URL vào session_state để giữ giá trị giữa các lần chạy lại ứng dụng
            st.session_state["ngrok_url"] = public_url

        except Exception as e:
            st.sidebar.error(f"❌ Lỗi khi kết nối Ngrok: {e}")
    else:
        st.sidebar.info("ℹ️ Nhập Ngrok Authtoken để tạo URL public. Bạn có thể lấy token miễn phí tại [ngrok.com](https://ngrok.com).")

# Tải dữ liệu
data = load_data()

# Tiền xử lý dữ liệu
if not data.empty:
    processed_data = preprocess_data(data)

    # Huấn luyện mô hình
    with st.spinner("Đang huấn luyện mô hình dự đoán giá..."):
        model, r2_score, rmse = train_model(processed_data)

    # Nếu không có dữ liệu, hiển thị thông báo
else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn đến file dữ liệu.")
    st.stop()

# Tạo sidebar
st.sidebar.title("🏠 Vietnam Real Estate")
app_mode = st.sidebar.selectbox("Chọn chế độ", ["Dự đoán giá", "Phân tích dữ liệu", "Về dự án"])

# Kết nối Ngrok nếu người dùng chọn
if st.sidebar.checkbox("Bật kết nối Ngrok", False):
    run_ngrok()

# Hiển thị thông tin trên sidebar
st.sidebar.subheader("Thông tin mô hình")
st.sidebar.metric("Độ chính xác (R²)", f"{r2_score:.4f}")
st.sidebar.metric("RMSE", f"{rmse:.4f}")
st.sidebar.metric("Số lượng bất động sản", f"{len(data):,}")

# Footer của sidebar
st.sidebar.markdown("---")
st.sidebar.info(
    "Dự án Dự đoán giá bất động sản Việt Nam sử dụng PySpark, Streamlit và Ngrok. "
    "Dữ liệu được thu thập từ nhadat.cafeland.vn."
)

# CHẾ ĐỘ 1: DỰ ĐOÁN GIÁ
if app_mode == "Dự đoán giá":
    st.title("🏘️ Dự đoán giá bất động sản Việt Nam")
    st.markdown("### Nhập thông tin về bất động sản để nhận dự đoán giá")

    # Tạo layout với 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📍 Vị trí")
        # Tạo card bằng cách dùng container với CSS tùy chỉnh
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Chọn tỉnh/thành phố
            city_options = sorted(data["city_province"].unique())
            city = st.selectbox("Tỉnh/Thành phố", city_options)

            # Lọc quận/huyện dựa trên tỉnh/thành phố đã chọn
            district_options = sorted(data[data["city_province"] == city]["district"].unique())
            district = st.selectbox("Quận/Huyện", district_options)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🏠 Đặc điểm bất động sản")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Thông tin cơ bản về BĐS
            area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=10.0)
            category_options = sorted(data["category"].unique())
            category = st.selectbox("Loại bất động sản", category_options)
            direction_options = sorted(data["direction"].unique())
            direction = st.selectbox("Hướng nhà", direction_options)
            liability_options = sorted(data["liability"].unique())
            liability = st.selectbox("Tình trạng pháp lý", liability_options)

            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🚪 Thông tin phòng ốc")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Thông tin phòng ốc
            bedroom_num = st.number_input("Số phòng ngủ", min_value=0, max_value=10, value=2, step=1)
            floor_num = st.number_input("Số tầng", min_value=0, max_value=50, value=2, step=1)
            toilet_num = st.number_input("Số nhà vệ sinh", min_value=0, max_value=10, value=2, step=1)
            livingroom_num = st.number_input("Số phòng khách", min_value=0, max_value=10, value=1, step=1)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### 🛣️ Thông tin khu vực")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Thông tin khu vực
            street_width = st.number_input("Chiều rộng đường (m)", min_value=0.0, max_value=50.0, value=8.0, step=0.5)

            st.markdown('</div>', unsafe_allow_html=True)

    # Nút dự đoán
    if st.button("Dự đoán giá", type="primary"):
        # Chuẩn bị dữ liệu đầu vào
        input_data = {
            "area_m2": area,
            "bedroom_num": bedroom_num,
            "floor_num": floor_num,
            "toilet_num": toilet_num,
            "livingroom_num": livingroom_num,
            "street_width_m": street_width,
            "city_province": city,
            "district": district,
            "category": category,
            "direction": direction,
            "liability": liability,
            # Các trường cần thiết cho mô hình
            "price_per_m2": 0,  # Giá trị này sẽ bị bỏ qua trong dự đoán
            "price_log": 0      # Giá trị này sẽ bị bỏ qua trong dự đoán
        }

        # Dự đoán giá
        with st.spinner("Đang dự đoán giá..."):
            try:
                # Thêm hiệu ứng chờ để cải thiện UX
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    time.sleep(0.1)  # Tạo độ trễ giả để hiệu ứng đẹp hơn
                    progress_bar.progress(percent_complete)
                progress_bar.empty()  # Xóa thanh tiến trình sau khi hoàn thành

                # Thực hiện dự đoán
                predicted_price_per_m2 = predict_price(model, input_data)
                total_price = predicted_price_per_m2 * area

                # Hiển thị kết quả trong container đẹp
                st.markdown("#### 📊 Kết quả dự đoán")
                with st.container():
                    st.markdown('<div class="card" style="background-color: #eaf7ea;">', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Giá dự đoán / m²", f"{predicted_price_per_m2:,.0f} VND")
                    with col2:
                        st.metric("Tổng giá dự đoán", f"{total_price:,.0f} VND")

                    # Hiển thị theo tỷ VND cho dễ đọc
                    total_price_billion = total_price / 1e9
                    st.info(f"💰 Tổng giá dự đoán: **{total_price_billion:.2f} tỷ VND**")

                    st.markdown('</div>', unsafe_allow_html=True)

                # Hiển thị các bất động sản tương tự
                st.markdown("#### 🔍 Bất động sản tương tự")
                similar_properties = data[
                    (data["city_province"] == city) &
                    (data["district"] == district) &
                    (data["area_m2"] > area * 0.7) &
                    (data["area_m2"] < area * 1.3)
                ]

                if len(similar_properties) > 0:
                    similar_df = similar_properties[["area_m2", "price_per_m2", "bedroom_num", "floor_num", "category"]].head(5).reset_index(drop=True)
                    similar_df.columns = ["Diện tích (m²)", "Giá/m² (VND)", "Số phòng ngủ", "Số tầng", "Loại BĐS"]
                    st.dataframe(similar_df, use_container_width=True)
                else:
                    st.info("Không tìm thấy bất động sản tương tự trong dữ liệu.")

            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

# CHẾ ĐỘ 2: PHÂN TÍCH DỮ LIỆU
elif app_mode == "Phân tích dữ liệu":
    st.title("📊 Phân tích dữ liệu bất động sản Việt Nam")

    # Tạo tabs để phân chia nội dung
    tab1, tab2, tab3 = st.tabs(["📈 Phân phối giá", "📍 Phân tích vị trí", "🏠 Đặc điểm bất động sản"])

    with tab1:
        st.subheader("Phân tích phân phối giá bất động sản")

        # Vẽ biểu đồ phân phối giá
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # Phân phối giá ban đầu
        sns.histplot(data["price_per_m2"], kde=True, ax=ax[0])
        ax[0].set_title("Phân phối giá / m²")
        ax[0].set_xlabel("Giá (VND/m²)")
        ax[0].set_ylabel("Số lượng")

        # Phân phối giá sau khi biến đổi log
        sns.histplot(np.log1p(data["price_per_m2"]), kde=True, ax=ax[1])
        ax[1].set_title("Phân phối logarit của giá / m²")
        ax[1].set_xlabel("ln(Giá/m²)")
        ax[1].set_ylabel("Số lượng")

        plt.tight_layout()
        st.pyplot(fig)

        # Tương tác: Lọc theo khoảng giá
        st.subheader("Lọc dữ liệu theo khoảng giá")

        # Tạo slider chọn khoảng giá
        price_range = st.slider(
            "Chọn khoảng giá (VND/m²)",
            min_value=float(data["price_per_m2"].min()),
            max_value=float(data["price_per_m2"].max()),
            value=(float(data["price_per_m2"].quantile(0.25)), float(data["price_per_m2"].quantile(0.75)))
        )

        # Lọc dữ liệu theo khoảng giá đã chọn
        filtered_data = data[(data["price_per_m2"] >= price_range[0]) & (data["price_per_m2"] <= price_range[1])]

        # Hiển thị thông tin về dữ liệu đã lọc
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số lượng BĐS", f"{len(filtered_data)}")
        with col2:
            st.metric("Giá trung bình/m²", f"{filtered_data['price_per_m2'].mean():,.0f} VND")
        with col3:
            st.metric("Diện tích trung bình", f"{filtered_data['area_m2'].mean():.1f} m²")

        # Hiển thị dữ liệu đã lọc
        st.dataframe(filtered_data[["city_province", "district", "area_m2", "price_per_m2", "category"]].head(10))

    with tab2:
        st.subheader("Phân tích giá theo vị trí địa lý")

        # Phân tích giá trung bình theo tỉnh/thành phố
        st.markdown("#### Giá trung bình theo tỉnh/thành phố")

        # Tính giá trung bình theo tỉnh/thành phố
        city_price = data.groupby("city_province")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        city_price.columns = ["Tỉnh/Thành phố", "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            city_price.head(10),
            x="Tỉnh/Thành phố",
            y="Giá trung bình/m²",
            title="Top 10 tỉnh/thành phố có giá bất động sản cao nhất",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Phân tích giá theo quận/huyện trong một tỉnh/thành phố đã chọn
        st.markdown("#### Giá trung bình theo quận/huyện")

        # Chọn tỉnh/thành phố để xem chi tiết
        selected_city = st.selectbox("Chọn tỉnh/thành phố", sorted(data["city_province"].unique()))

        # Lọc dữ liệu theo tỉnh/thành phố đã chọn
        city_data = data[data["city_province"] == selected_city]

        # Tính giá trung bình theo quận/huyện
        district_price = city_data.groupby("district")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        district_price.columns = ["Quận/Huyện", "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            district_price,
            x="Quận/Huyện",
            y="Giá trung bình/m²",
            title=f"Giá bất động sản trung bình theo quận/huyện tại {selected_city}",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Phân tích mối quan hệ giữa đặc điểm và giá")

        # Biểu đồ phân tán: Diện tích vs. Giá
        st.markdown("#### Mối quan hệ giữa diện tích và giá")

        # Tạo mẫu nhỏ hơn nếu có quá nhiều dữ liệu
        sample_size = min(1000, len(data))
        sampled_data = data.sample(sample_size, random_state=42)

        # Vẽ biểu đồ phân tán
        fig = px.scatter(
            sampled_data,
            x="area_m2",
            y="price_per_m2",
            color="city_province",
            size="bedroom_num",
            hover_data=["district", "category"],
            title="Mối quan hệ giữa diện tích và giá",
            labels={
                "area_m2": "Diện tích (m²)",
                "price_per_m2": "Giá/m² (VND)",
                "city_province": "Tỉnh/Thành phố",
                "bedroom_num": "Số phòng ngủ"
            },
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Ma trận tương quan
        st.markdown("#### Ma trận tương quan giữa các đặc điểm số")

        # Chọn các đặc trưng số để tính tương quan
        numeric_features = ["area_m2", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street_width_m", "price_per_m2"]
        corr_matrix = data[numeric_features].corr()

        # Vẽ heatmap tương quan
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        plt.title("Ma trận tương quan giữa các đặc điểm")
        st.pyplot(fig)

        # Phân tích theo đặc điểm bất động sản
        st.markdown("#### Phân tích giá theo đặc điểm")

        # Chọn đặc điểm để phân tích
        feature = st.selectbox(
            "Chọn đặc điểm",
            ["category", "direction", "liability", "bedroom_num", "floor_num"]
        )

        # Tính giá trung bình theo đặc điểm đã chọn
        if feature in ["bedroom_num", "floor_num"]:
            # Đối với đặc điểm số, chuyển đổi thành chuỗi để nhóm
            data["feature_str"] = data[feature].astype(str)
            feature_price = data.groupby("feature_str")["price_per_m2"].mean().reset_index()
            feature_price.columns = [feature, "Giá trung bình/m²"]

            # Sắp xếp theo thứ tự số
            feature_price[feature] = feature_price[feature].astype(float)
            feature_price = feature_price.sort_values(by=feature)
            feature_price[feature] = feature_price[feature].astype(str)
        else:
            # Đối với đặc điểm phân loại
            feature_price = data.groupby(feature)["price_per_m2"].mean().sort_values(ascending=False).reset_index()
            feature_price.columns = [feature, "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            feature_price,
            x=feature,
            y="Giá trung bình/m²",
            title=f"Giá trung bình theo {feature}",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

# CHẾ ĐỘ 3: VỀ DỰ ÁN
else:
    st.title("ℹ️ Về dự án dự đoán giá bất động sản Việt Nam")

    # Giới thiệu dự án
    st.markdown("""
    ## 🏠 Dự đoán giá bất động sản Việt Nam

    Đây là ứng dụng demo cho mô hình dự đoán giá bất động sản tại Việt Nam sử dụng học máy.
    Ứng dụng được phát triển như một phần của dự án nghiên cứu về ứng dụng dữ liệu lớn
    trong phân tích thị trường bất động sản.

    ### 🛠️ Công nghệ sử dụng

    - **Thu thập dữ liệu**: Selenium, BeautifulSoup
    - **Xử lý dữ liệu lớn**: Apache Spark (PySpark)
    - **Học máy**: Gradient Boosted Trees, Random Forest, Linear Regression
    - **Giao diện người dùng**: Streamlit
    - **Triển khai**: Ngrok

    ### 📊 Bộ dữ liệu

    Bộ dữ liệu gồm thông tin về hơn {len(data):,} bất động sản được thu thập từ website nhadat.cafeland.vn, bao gồm:

    - Vị trí (tỉnh/thành phố, quận/huyện)
    - Diện tích, số phòng, số tầng
    - Đặc điểm bất động sản (loại bất động sản, hướng nhà...)
    - Giá/m²

    ### 📝 Quy trình xử lý dữ liệu

    1. **Thu thập dữ liệu**: Sử dụng web scraping để thu thập dữ liệu từ trang bất động sản
    2. **Làm sạch dữ liệu**: Xử lý giá trị thiếu, định dạng lại các trường, xử lý ngoại lệ
    3. **Kỹ thuật đặc trưng**: Tạo các đặc trưng mới, mã hóa đặc trưng phân loại
    4. **Huấn luyện mô hình**: Sử dụng Gradient Boosted Trees để dự đoán giá
    5. **Đánh giá mô hình**: R² = {r2_score:.4f}, RMSE = {rmse:.4f}

    ### 👥 Nhóm phát triển

    Dự án được phát triển bởi nhóm sinh viên ngành Data Science:

    - MSSV: 1234567
    - MSSV: 1234568
    - MSSV: 1234569

    ### 📱 Hướng dẫn sử dụng

    - Sử dụng thanh điều hướng bên trái để chuyển đổi giữa các chế độ
    - Trong phần "Dự đoán giá", nhập thông tin bất động sản để nhận dự đoán
    - Trong phần "Phân tích dữ liệu", khám phá các xu hướng và mẫu trong dữ liệu
    - Sử dụng tính năng Ngrok để chia sẻ ứng dụng với người khác
    """)

    # Thêm hình ảnh minh họa
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png", width=300)

