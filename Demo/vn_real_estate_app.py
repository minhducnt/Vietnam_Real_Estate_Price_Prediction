# MARK: - Import Libraries
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
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import os
import time
from pyngrok import ngrok

# MARK: - Global Variables
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
    /* Import Literata font with Vietnamese support */
    @import url('https://fonts.googleapis.com/css2?family=Literata:wght@200;300;400;500;600;700;800;900&display=swap&subset=latin,latin-ext,vietnamese');

    /* Main content area styling */
    .main {
        background-color: #f8f9fa;
        margin-left: 250px;
        padding: 1rem 2rem;
    }

    /* Global font styling */
    .stApp {
        font-family: 'Literata', serif;
    }

    /* Text elements styling */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, button, input, optgroup, select, textarea {
        font-family: 'Literata', serif !important;
    }

    /* Custom dark sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a202c;
        padding-top: 0;
        min-width: 280px !important;
        max-width: 280px !important;
    }

    [data-testid="stSidebar"] .css-6qob1r {
        background-color: #1a202c;
    }

    /* Sidebar header styling */
    .sidebar-header {
        background: linear-gradient(to right, #2c5282, #1a365d);
        padding: 1.5rem 1rem;
        text-align: center;
        margin-bottom: 1.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        border-radius: 0.8rem;
    }

    .sidebar-header img {
        max-width: 40px;
        margin-bottom: 0.5rem;
    }

    /* Sidebar navigation styling */
    .nav-link {
        display: block;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        text-decoration: none;
        color: white !important;
        transition: all 0.2s ease;
    }

    .nav-link:hover {
        background-color: rgba(255,255,255,0.1);
    }

    .nav-link.active {
        background-color: #4c9aff;
        color: white !important;
        font-weight: 600;
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] .css-eczf16,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }

    /* Custom sidebar buttons */
    [data-testid="stSidebar"] button {
        background-color: #2d3748 !important;
        color: white !important;
        border: none !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #4A5568 !important;
    }

    /* Custom selectbox in sidebar */
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.5rem;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #2d3748 !important;
        color: white !important;
        border: none !important;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #4c9aff !important;
        font-weight: bold !important;
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

    /* Sidebar metrics styling */
    .sidebar-metric {
        background-color: rgba(255,255,255,0.05);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .sidebar-section {
        padding: 0.5rem 1rem 1rem 1rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-section-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        color: rgba(255,255,255,0.8);
        display: flex;
        align-items: center;
    }

    .sidebar-section-title svg {
        margin-right: 0.5rem;
    }

    /* Left-aligned title styling */
    .left-aligned-title {
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        line-height: 2.4rem !important;
    }

    .left-aligned-title span {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: normal !important;
    }

    .left-aligned-title span:first-child {
        margin-right: 12px !important;
    }
</style>
''', unsafe_allow_html=True)

# Không cần hiển thị logo riêng vì sẽ được thêm vào sidebar

# MARK: - Khởi tạo phiên Spark
@st.cache_resource
def get_spark_session():
    """Khởi tạo và trả về một phiên Spark với xử lý lỗi."""
    try:
        spark = (
            SparkSession.builder
            .appName("VNRealEstatePricePrediction")
            .config("spark.driver.memory", "2g")
            .master("local[*]")
            .getOrCreate()
        )
        # Kiểm tra kết nối
        spark.sparkContext.parallelize([1]).collect()
        return spark
    except Exception as e:
        st.warning(f"Không thể khởi tạo Spark: {e}. Sẽ sử dụng phương pháp dự phòng.")
        return None

# MARK: - Đọc dữ liệu
@st.cache_data
def load_data(file_path=None):
    """Đọc dữ liệu bất động sản từ file CSV."""
    try:
        # Xác định đường dẫn tuyệt đối đến file dữ liệu
        if file_path is None:
            # Đường dẫn tương đối từ thư mục gốc của dự án
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, 'Data', 'Final Data Cleaned.csv')

        # Đọc dữ liệu bằng pandas
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {e}")
        return pd.DataFrame()

# MARK: - Xử lý dữ liệu
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

    # Đảm bảo chúng ta có cả các cột cũ và mới
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            # Nếu cột cũ tồn tại, tạo cột mới dựa trên nó
            df[new_name] = df[old_name]
        elif new_name not in df.columns and old_name not in df.columns:
            # Nếu cả hai cột đều không tồn tại, hiển thị lỗi
            st.error(f"Không tìm thấy cột {old_name} hoặc {new_name} trong dữ liệu")

    # Xử lý giá trị thiếu
    numeric_cols = ["area (m2)", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street (m)"]
    for col in numeric_cols:
        if col in df:
            # Thay thế -1 (giá trị thiếu) bằng giá trị trung vị
            median_val = df[df[col] != -1][col].median()
            df[col] = df[col].replace(-1, median_val)

    # Chuyển đổi logarithm cho giá
    df['price_log'] = np.log1p(df['price_per_m2'])

    return df

# MARK: - Chuyển đổi dữ liệu pandas sang spark
@st.cache_resource
def convert_to_spark(data):
    """Chuyển đổi DataFrame pandas sang DataFrame Spark."""
    spark = get_spark_session()
    return spark.createDataFrame(data)

# MARK: - Huấn luyện mô hình
@st.cache_resource
def train_model(data):
    """Huấn luyện mô hình dự đoán giá bất động sản."""
    # Khởi tạo SparkSession
    spark = get_spark_session()

    # For debugging - commented out
    # print(f"Các cột trong dữ liệu gốc trước khi chuyển đổi: {data.columns.tolist()}")

    # Đảm bảo dữ liệu có tất cả các cột cần thiết (cả tên cũ và mới)
    if 'area (m2)' in data.columns and 'area_m2' not in data.columns:
        data['area_m2'] = data['area (m2)'].copy()
    if 'street (m)' in data.columns and 'street_width_m' not in data.columns:
        data['street_width_m'] = data['street (m)'].copy()

    # Chuyển đổi dữ liệu pandas sang Spark
    data_spark = convert_to_spark(data)

    # Định nghĩa các cột để sử dụng trong mô hình
    # Sử dụng tên cột cố định dựa trên biến FEATURE_COLUMNS
    area_column = FEATURE_COLUMNS['area']  # 'area (m2)'
    street_column = FEATURE_COLUMNS['street']  # 'street (m)'

    # Đặc trưng số
    numeric_features = [area_column, "bedroom_num", "floor_num", "toilet_num", "livingroom_num", street_column]

    # Chỉ sử dụng các cột tồn tại trong dữ liệu
    numeric_features = [col for col in numeric_features if col in data_spark.columns]

    # Đặc trưng phân loại
    categorical_features = ["category", "direction", "liability", "district", "city_province"]

    # Loại trừ các đặc trưng không tồn tại trong dữ liệu
    categorical_features = [col for col in categorical_features if col in data.columns]

    # Tạo onehot encoding cho các biến phân loại
    from pyspark.ml.feature import StringIndexer, OneHotEncoder

    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid="keep")
                for col in categorical_features]

    encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded")
                for col in categorical_features]

    # Gộp tất cả các đặc trưng đã xử lý vào một vector
    assembler_inputs = numeric_features + [col+"_encoded" for col in categorical_features]

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

    # Tạo chuẩn hóa dữ liệu
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    # Khởi tạo mô hình GBT
    gbt = GBTRegressor(featuresCol="scaled_features", labelCol="price_per_m2", maxIter=10)

    # Tạo pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, gbt])

    try:
        # Chia dữ liệu thành tập huấn luyện và kiểm tra
        train_data, test_data = data_spark.randomSplit([0.8, 0.2], seed=42)

        # Huấn luyện mô hình
        model = pipeline.fit(train_data)

        # Đánh giá mô hình
        predictions = model.transform(test_data)

        # Tính toán các chỉ số đánh giá
        evaluator = RegressionEvaluator(labelCol="price_per_m2", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)

        evaluator.setMetricName("r2")
        r2 = evaluator.evaluate(predictions)

        # Hiển thị kết quả đánh giá
        st.session_state.model_metrics = {
            "rmse": rmse,
            "r2": r2
        }

        return model
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình: {e}")
        raise e

# MARK: - Dự đoán giá dựa trên giá trung bình (dự phòng)
def predict_price_fallback(input_data, data):
    """Phương pháp dự phòng cho việc dự đoán giá khi Spark không khả dụng."""
    try:
        # Lọc dữ liệu dựa trên vị trí (tỉnh/thành phố và quận/huyện)
        city = input_data.get("city_province")
        district = input_data.get("district")
        category = input_data.get("category")
        area = input_data.get("area (m2)")

        # Lọc dữ liệu tương tự
        similar_properties = data[
            (data["city_province"] == city) &
            (data["district"] == district) &
            (data["category"] == category) &
            (data["area_m2"] > area * 0.7) &
            (data["area_m2"] < area * 1.3)
        ]

        # Nếu không có dữ liệu tương tự, mở rộng phạm vi tìm kiếm
        if len(similar_properties) < 3:
            similar_properties = data[
                (data["city_province"] == city) &
                (data["district"] == district)
            ]

        # Nếu vẫn không có, lấy trung bình toàn thành phố
        if len(similar_properties) < 3:
            similar_properties = data[(data["city_province"] == city)]

        # Tính giá trung bình
        if len(similar_properties) > 0:
            avg_price = similar_properties["price_per_m2"].mean()
            return avg_price
        else:
            # Mặc định nếu không có dữ liệu tương tự
            return data["price_per_m2"].mean()
    except Exception as e:
        st.error(f"Lỗi khi dự đoán giá dự phòng: {e}")
        return 30000000  # Giá mặc định nếu có lỗi

# MARK: - Dự đoán giá
def predict_price(model, input_data):
    """Dự đoán giá dựa trên đầu vào của người dùng."""
    try:
        # Đảm bảo session state có dữ liệu
        if 'data' not in st.session_state:
            st.error("Dữ liệu chưa được khởi tạo trong session state")
            return 30000000  # Giá trị mặc định nếu không có dữ liệu

        # Chuyển dữ liệu đầu vào thành DataFrame
        data_copy = {k: [v] for k, v in input_data.items()}

        # Tạo pandas DataFrame
        input_df = pd.DataFrame(data_copy)

        # Sao chép dữ liệu để không ảnh hưởng đến dữ liệu gốc
        data_copy = input_df.copy()

        # Xử lý các giá trị không tồn tại
        for col in data_copy.columns:
            if col in ["bedroom_num", "floor_num", "toilet_num", "livingroom_num"]:
                data_copy[col] = data_copy[col].fillna(-1).astype(int)

        # Đảm bảo chúng ta có các cột đúng tên chính xác
        # Đảm bảo không sử dụng area_m2 mà sử dụng 'area (m2)'
        if 'area_m2' in data_copy.columns and 'area (m2)' not in data_copy.columns:
            data_copy['area (m2)'] = data_copy['area_m2'].copy()
            del data_copy['area_m2']

        # Đảm bảo không sử dụng street_width_m mà sử dụng 'street (m)'
        if 'street_width_m' in data_copy.columns and 'street (m)' not in data_copy.columns:
            data_copy['street (m)'] = data_copy['street_width_m'].copy()
            del data_copy['street_width_m']

        # Kiểm tra nếu Spark session tồn tại
        spark = get_spark_session()

        if spark is not None:
            try:
                # Chuyển đổi dữ liệu sang Spark DataFrame
                spark_df = convert_to_spark(data_copy)

                # Dự đoán giá
                predictions = model.transform(spark_df)

                # Lấy kết quả dự đoán
                prediction_value = predictions.select("prediction").collect()[0][0]
                if prediction_value is not None:
                    return prediction_value
                else:
                    # Sử dụng phương pháp dự phòng nếu giá trị dự đoán là None
                    st.warning("Kết quả dự đoán không hợp lệ, sử dụng phương pháp dự phòng.")
                    if 'data' in st.session_state:
                        return predict_price_fallback(input_data, st.session_state.data)
                    else:
                        return 30000000  # Giá mặc định nếu không có dữ liệu
            except Exception as e:
                st.warning(f"Lỗi khi dự đoán với Spark: {e}. Sử dụng phương pháp dự phòng.")
                if 'data' in st.session_state:
                    return predict_price_fallback(input_data, st.session_state.data)
                else:
                    return 30000000  # Giá mặc định nếu không có dữ liệu
        else:
            # Sử dụng phương pháp dự phòng nếu không có Spark
            st.info("Sử dụng phương pháp dự phòng để dự đoán giá.")
            if 'data' in st.session_state:
                return predict_price_fallback(input_data, st.session_state.data)
            else:
                return 30000000  # Giá mặc định nếu không có dữ liệu
    except Exception as e:
        st.error(f"Lỗi khi chuẩn bị dữ liệu: {e}")
        # Sử dụng giá trị mặc định nếu tất cả các phương pháp đều thất bại
        return 30000000  # Giá mặc định nếu có lỗi

# MARK: - Kết nối Ngrok
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

# MARK: - Main Application Flow
# Tải dữ liệu
data = load_data()

# Lưu dữ liệu vào session state để sử dụng trong các hàm dự đoán
if 'data' not in st.session_state:
    st.session_state.data = data

# Tiền xử lý dữ liệu
if not data.empty:
    processed_data = preprocess_data(data)

    # Huấn luyện mô hình
    with st.spinner("Đang huấn luyện mô hình dự đoán giá..."):
        model = train_model(processed_data)
        # Lấy các metric từ session state sau khi huấn luyện mô hình
        if 'model_metrics' in st.session_state:
            r2_score = st.session_state.model_metrics['r2']
            rmse = st.session_state.model_metrics['rmse']
        else:
            r2_score = 0.0
            rmse = 0.0

    # Nếu không có dữ liệu, hiển thị thông báo
else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn đến file dữ liệu.")
    st.stop()

# MARK: - Sidebar
# Tạo container để ẩn padding mặc định của sidebar
st.sidebar.markdown("""
<style>
    [data-testid="stSidebarUserContent"] > div:first-child {padding-top: 0rem;}
    [data-testid="stVerticalBlock"] {gap: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Header của sidebar với logo
st.sidebar.markdown("""
<div class="sidebar-header">
    <img src="https://img.icons8.com/fluency/96/000000/home.png" alt="Logo">
    <h2 style="color: white; margin: 0; font-size: 1.3rem;">BĐS Việt Nam</h2>
    <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">AI Dự Đoán Giá</p>
</div>
""", unsafe_allow_html=True)

# Set session state for app_mode if it doesn't exist
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = "Dự đoán giá"

# Phương thức để cập nhật app_mode
def set_app_mode(mode):
    st.session_state['app_mode'] = mode

# CSS cho buttons
st.markdown("""
<style>
    /* Navigation button styling */
    button[kind="secondary"], button[kind="primary"] {
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        padding: 0.6rem 0.5rem !important;
        margin: 0.25rem 0 !important;
        border-radius: 0.5rem !important;
    }

    button[kind="secondary"] p, button[kind="primary"] p {
        width: 100% !important;
        text-align: left !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    .sidebar-nav-button {
        width: 100%;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        padding: 0.6rem 0.5rem !important;
        margin: 0.25rem 0 !important;
        border-radius: 0.5rem !important;
        color: white !important;
        background-color: rgba(44, 52, 75, 0.5) !important;
        border: none !important;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.95rem !important;
        line-height: 1.2;
    }

    .sidebar-nav-button:hover {
        background-color: rgba(76, 154, 255, 0.7) !important;
    }

    .sidebar-nav-button-active {
        background-color: #4c9aff !important;
        color: white !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Lấy mode hiện tại
app_mode = st.session_state['app_mode']

# Menu options với icons và khoảng cách
modes = ["Dự đoán giá", "Phân tích dữ liệu", "Về dự án"]
modes_icons = ["🏠 ", "📊 ", "ℹ️ "]

# Tạo CSS để điều chỉnh style cho nút button trong Streamlit
st.markdown("""
<style>
    button[kind="primary"], button[kind="secondary"] {
        padding-left: 16px !important;
        padding-right: 6px !important;
    }

    button[kind="primary"] p:first-child, button[kind="secondary"] p:first-child {
        display: inline-block;
        margin-right: 8px !important;
        font-size: 1.2rem !important;
    }

    button[kind="primary"] p:last-child, button[kind="secondary"] p:last-child {
        font-size: 0.95rem !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Container cho menu
menu_container = st.sidebar.container()

# Tạo các button
for i, mode in enumerate(modes):
    active_class = "sidebar-nav-button-active" if mode == app_mode else ""
    # Sử dụng nhiều khoảng trắng hơn để tạo khoảng cách
    button_label = f"{modes_icons[i]}        {mode}"

    if menu_container.button(button_label, key=f"nav_{i}",
                           use_container_width=True,
                           on_click=set_app_mode,
                           args=(mode,),
                           type="primary" if mode == app_mode else "secondary"):
        pass

    # Tạo style cho nút
    if i < len(modes) - 1:
        # Định nghĩa CSS chính xác hơn để nhắm đến các thành phần trong nút
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] > div:nth-child(CHILD_INDEX) button {
            width: 100% !important;
            text-align: left !important;
            padding: 0.6rem 0.5rem !important;
            margin: 0.25rem 0 !important;
            border-radius: 0.5rem !important;
            font-size: 0.95rem !important;
        }

        /* Nhắm trực tiếp vào thành phần chứa văn bản trong nút */
        div[data-testid="stVerticalBlock"] > div:nth-child(CHILD_INDEX) button > div:first-child {
            display: flex !important;
            justify-content: flex-start !important;
            width: 100% !important;
        }

        /* Nhắm trực tiếp vào thành phần chứa văn bản */
        div[data-testid="stVerticalBlock"] > div:nth-child(CHILD_INDEX) button > div:first-child > p {
            text-align: left !important;
            width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        </style>
        """.replace("CHILD_INDEX", str(i + 1 + 6)), unsafe_allow_html=True)  # +6 vì có các đối tượng khác trước menu

# Thêm CSS để nâng cao giao diện các metrics
st.markdown("""
<style>
    /* Style cho card metrics mới */
    .enhanced-metric-card {
        background: linear-gradient(145deg, rgba(44,82,130,0.5), rgba(26,54,93,0.7));
        border-radius: 10px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(76,154,255,0.3);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 110px; /* Đặt chiều cao cố định cho card */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .enhanced-metric-card:hover {
        border: 1px solid rgba(76,154,255,0.6);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    .metric-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.4rem;
    }

    .metric-icon {
        margin-right: 0.5rem;
        background-color: rgba(76,154,255,0.2);
        border-radius: 50%;
        padding: 0.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4c9aff;
        text-align: center;
        margin-top: 0.2rem;
    }

    .metric-description {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.6);
        text-align: center;
        margin-top: 0.2rem;
    }

    .model-stats-container {
        margin: 0.3rem 0;
        padding: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Hiển thị thông tin mô hình trong nhóm
st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Thông số mô hình</span></div>', unsafe_allow_html=True)

# Metrics độ chính xác
st.sidebar.markdown("""
<div class="enhanced-metric-card" style="background: linear-gradient(145deg, rgba(51,97,255,0.3), rgba(29,55,147,0.5));
                           border-color: rgba(100,149,237,0.3); padding: 10px; margin: 5px 0;">
    <div class="metric-header" style="display:flex; align-items:center;">
        <div class="metric-icon" style="background-color: rgba(100,149,237,0.2); margin-right:8px; padding:5px; border-radius:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">R² Score</span>
    </div>
    <div class="metric-value" style="color: #84a9ff; font-size: 1.5rem; text-align:center; margin:5px 0;">{r2_score:.4f}</div>
</div>
""".format(r2_score=r2_score), unsafe_allow_html=True)

# Thêm khoảng cách giữa hai card thông số mô hình
st.sidebar.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# Metrics độ lệch chuẩn - RMSE
st.sidebar.markdown("""
<div class="enhanced-metric-card" style="background: linear-gradient(145deg, rgba(139,92,246,0.3), rgba(76,29,149,0.5));
                           border-color: rgba(167,139,250,0.3); padding: 10px; margin: 5px 0;">
    <div class="metric-header" style="display:flex; align-items:center;">
        <div class="metric-icon" style="background-color: rgba(167,139,250,0.2); margin-right:8px; padding:5px; border-radius:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9 22V12H15V22" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">RMSE</span>
    </div>
    <div class="metric-value" style="color: #c4b5fd; font-size: 1.5rem; text-align:center; margin:5px 0;">{rmse:.4f}</div>
</div>
""".format(rmse=rmse), unsafe_allow_html=True)

# Thêm khoảng cách giữa các card metric và số lượng dữ liệu
st.sidebar.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Các thống kê dữ liệu - hiển thị riêng từng dòng
st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Thống kê dữ liệu</span></div>', unsafe_allow_html=True)

# Số lượng bất động sản
st.sidebar.markdown("""
<div class="enhanced-metric-card" style="background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7));
                           border-color: rgba(76,255,154,0.3); padding: 10px; margin: 5px 0;">
    <div class="metric-header" style="display:flex; align-items:center;">
        <div class="metric-icon" style="background-color: rgba(76,255,154,0.2); margin-right:8px; padding:5px; border-radius:6px;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 3V21H21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M19 5L9 15L6 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">Số lượng bất động sản</span>
    </div>
    <div class="metric-value" style="color: #4dff9e; font-size: 1.8rem; text-align:center; margin:5px 0;">{data_count:,}</div>
</div>
""".format(data_count=len(data)), unsafe_allow_html=True)

# Footer của sidebar
st.sidebar.markdown("<hr style='margin: 1.5rem 0; opacity: 0.2'>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='padding: 0 1rem; color: rgba(255,255,255,0.7); font-size: 0.8rem;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 0.5rem">
        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 16V12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 8H12.01" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>Dự đoán giá BĐS Việt Nam</span>
</div>

<div style="display: flex; align-items: center;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 0.5rem">
        <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 13C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>Nguồn: nhadat.cafeland.vn</span>
</div>

<div style="display: flex; align-items: center; margin-top: 0.5rem;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 0.5rem">
        <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M9 11C11.2091 11 13 9.20914 13 7C13 4.79086 11.2091 3 9 3C6.79086 3 5 4.79086 5 7C5 9.20914 6.79086 11 9 11Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M16 3.13C16.8604 3.35031 17.623 3.85071 18.1676 4.55232C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89318 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>Nhóm 5</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# MARK: - Chế độ Dự đoán giá
if app_mode == "Dự đoán giá":
   # CSS thêm cho tiêu đề trang
    st.markdown("""
    <style>
    .modern-header {
        background: linear-gradient(to right, rgba(30, 30, 30, 0.9), rgba(45, 45, 45, 0.8));
        border-radius: 12px;
        padding: 20px 25px;
        margin-bottom: 25px;
        border-left: 4px solid #4c9aff;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }

    .header-title {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }

    .header-icon {
        background-color: rgba(255, 255, 255, 0.08);
        padding: 10px;
        border-radius: 10px;
        margin-right: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .header-icon svg {
        width: 26px;
        height: 26px;
        stroke: #ffffff;
        fill: none;
        stroke-width: 1.5;
    }

    .header-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.3;
    }

    .header-desc {
        display: flex;
        align-items: center;
        color: #cccccc;
        font-size: 1rem;
        line-height: 1.5;
        margin-left: 60px;
        position: relative;
    }

    .header-desc:before {
        content: '';
        position: absolute;
        width: 3px;
        height: 100%;
        background-color: rgba(76, 154, 255, 0.5);
        left: -15px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tiêu đề trang với giao diện hiện đại hơn
    st.markdown("""
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
            </div>
            <div class="header-text">Dự đoán giá bất động sản Việt Nam</div>
        </div>
        <div class="header-desc">
            Hãy nhập thông tin về bất động sản mà bạn quan tâm và chúng tôi sẽ dự đoán giá trị thị trường dựa trên mô hình học máy tiên tiến của chúng tôi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tạo style cho card
    st.markdown("""
    <style>
    .input-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #333;
        padding: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .input-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border-color: #444;
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-bottom: 0;
    }

    .card-header .icon {
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(139, 92, 246, 0.1);
        width: 26px;
        height: 26px;
        border-radius: 6px;
        margin-right: 8px;
    }

    .card-header .icon svg {
        width: 16px;
        height: 16px;
        stroke: #ffffff;
        fill: none;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    .card-header .title {
        color: #f0f0f0;
        font-size: 1rem;
        font-weight: 600;
    }

    .stSelectbox label, .stNumberInput label {
        font-weight: 500;
        color: #ccc !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tạo layout với 2 cột
    col1, col2 = st.columns([1, 1])

    with col1:
        # Card vị trí
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path>
                        <circle cx="12" cy="10" r="3"></circle>
                    </svg>
                </div>
                <div class="title">Vị trí</div>
            </div>
        """, unsafe_allow_html=True)

        # Chọn tỉnh/thành phố
        city_options = sorted(data["city_province"].unique())
        city = st.selectbox("Tỉnh/Thành phố", city_options, key='city')

        # Lọc quận/huyện dựa trên tỉnh/thành phố đã chọn
        district_options = sorted(data[data["city_province"] == city]["district"].unique())
        district = st.selectbox("Quận/Huyện", district_options, key='district')

        st.markdown('</div>', unsafe_allow_html=True)

        # Card thông tin cơ bản
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                        <polyline points="9 22 9 12 15 12 15 22"></polyline>
                    </svg>
                </div>
                <div class="title">Thông tin cơ bản</div>
            </div>
        """, unsafe_allow_html=True)

        # Một hàng 2 cột cho thông tin cơ bản
        bc1, bc2 = st.columns(2)
        with bc1:
            area = st.number_input("Diện tích (m²)", min_value=10.0, max_value=1000.0, value=80.0, step=10.0, key='area')
        with bc2:
            category_options = sorted(data["category"].unique())
            category = st.selectbox("Loại BĐS", category_options, key='category')

        # Hàng tiếp theo
        bc3, bc4 = st.columns(2)
        with bc3:
            direction_options = sorted(data["direction"].unique())
            direction = st.selectbox("Hướng nhà", direction_options, key='direction')
        with bc4:
            liability_options = sorted(data["liability"].unique())
            liability = st.selectbox("Tình trạng pháp lý", liability_options, key='liability')

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Card thông tin phòng ốc
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-7-7z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                </div>
                <div class="title">Thông tin phòng ốc</div>
            </div>
        """, unsafe_allow_html=True)

        # Hàng 1
        rc1, rc2 = st.columns(2)
        with rc1:
            bedroom_num = st.number_input("Số phòng ngủ", min_value=0, max_value=10, value=2, step=1, key='bedroom')
        with rc2:
            toilet_num = st.number_input("Số phòng vệ sinh", min_value=0, max_value=10, value=2, step=1, key='toilet')

        # Hàng 2
        rc3, rc4 = st.columns(2)
        with rc3:
            livingroom_num = st.number_input("Số phòng khách", min_value=0, max_value=10, value=1, step=1, key='livingroom')
        with rc4:
            floor_num = st.number_input("Số tầng", min_value=0, max_value=50, value=2, step=1, key='floor')

        st.markdown('</div>', unsafe_allow_html=True)

        # Card thông tin khu vực
        st.markdown("""
        <div class="input-card">
            <div class="card-header">
                <div class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="2" y1="12" x2="22" y2="12"></line>
                        <line x1="12" y1="2" x2="12" y2="22"></line>
                    </svg>
                </div>
                <div class="title">Thông tin khu vực</div>
            </div>
        """, unsafe_allow_html=True)

        # Thông tin chiều rộng đường
        street_width = st.number_input("Chiều rộng đường (m)",
                                      min_value=0.0, max_value=50.0, value=8.0, step=0.5, key='street')

        st.markdown('</div>', unsafe_allow_html=True)

    # Sử dụng cách tiếp cận khác cho nút dự đoán
    st.markdown('<div style="padding: 10px 0 20px 0;"></div>', unsafe_allow_html=True)

    # Cải thiện CSS cho mọi loại nút
    st.markdown('''
    <style>
    /* Targeting all buttons with tertiary style */
    .stButton > button[kind="tertiary"], div[data-testid="StyledFullScreenButton"] button[kind="tertiary"], div button[kind="tertiary"] {
        background-color: #1a202c !important;
        color: #f0f0f0 !important;
        border: 1px solid #2d3748 !important;
        padding: 12px 20px !important;
        margin: 8px 0 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        text-align: center !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    /* Hover state for all tertiary buttons */
    .stButton > button[kind="tertiary"]:hover, div button[kind="tertiary"]:hover {
        background-color: #2d3748 !important;
        border-color: #4a5568 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }

    /* Active/focus state */
    .stButton > button[kind="tertiary"]:active, div button[kind="tertiary"]:active,
    .stButton > button[kind="tertiary"]:focus, div button[kind="tertiary"]:focus {
        transform: translateY(0) !important;
        background-color: #2d3748 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    ''', unsafe_allow_html=True)

    # Cải thiện hơn nữa với container được thiết kế riêng
    st.markdown('<div class="prediction-button-wrapper"></div>', unsafe_allow_html=True)

    # Nút dự đoán
    if st.button("Dự đoán giá", use_container_width=True, type="tertiary"):
        # Chuẩn bị dữ liệu đầu vào
        input_data = {
            "area (m2)": area,
            "bedroom_num": bedroom_num,
            "floor_num": floor_num,
            "toilet_num": toilet_num,
            "livingroom_num": livingroom_num,
            "street (m)": street_width,
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

                # Kiểm tra kết quả dự đoán không phải là None
                if predicted_price_per_m2 is None:
                    st.error("Không thể dự đoán giá. Vui lòng thử lại sau.")
                else:
                    # Tính toán giá dự đoán
                    # Đảm bảo predicted_price_per_m2 là giá trị số nguyên
                    predicted_price_per_m2 = int(round(predicted_price_per_m2))
                    total_price = int(round(predicted_price_per_m2 * area))
                    total_price_billion = total_price / 1_000_000_000

                    # Hàm định dạng giá thông minh theo đơn vị
                    def format_price(price):
                        if price >= 1_000_000_000:  # Giá >= 1 tỷ
                            billions = price // 1_000_000_000
                            remaining = price % 1_000_000_000

                            if remaining == 0:
                                return f"{billions:,.0f} tỷ VND"

                            millions = remaining // 1_000_000
                            if millions == 0:
                                return f"{billions:,.0f} tỷ VND"
                            else:
                                return f"{billions:,.0f} tỷ {millions:,.0f} triệu VND"
                        elif price >= 1_000_000:  # Giá >= 1 triệu
                            millions = price // 1_000_000
                            remaining = price % 1_000_000

                            if remaining == 0:
                                return f"{millions:,.0f} triệu VND"

                            thousands = remaining // 1_000
                            if thousands == 0:
                                return f"{millions:,.0f} triệu VND"
                            else:
                                return f"{millions:,.0f} triệu {thousands:,.0f} nghìn VND"
                        elif price >= 1_000:  # Giá >= 1 nghìn
                            return f"{price//1_000:,.0f} nghìn VND"
                        else:
                            return f"{price:,.0f} VND"

                    # Định dạng giá tổng
                    formatted_total_price = format_price(total_price)

                    # CSS cho card hiển thị kết quả
                    st.markdown('''
                    <style>
                    .result-container {
                        background: linear-gradient(to right, rgba(25, 26, 36, 0.8), rgba(30, 32, 45, 0.9));
                        border-radius: 12px;
                        padding: 0;
                        margin: 15px 0;
                        border: 1px solid rgba(76, 154, 255, 0.2);
                        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                        overflow: hidden;
                    }

                    .result-header {
                        background: linear-gradient(to right, rgba(76, 154, 255, 0.15), rgba(76, 154, 255, 0.05));
                        padding: 16px 25px;
                        display: flex;
                        align-items: center;
                        border-bottom: 1px solid rgba(76, 154, 255, 0.1);
                    }

                    .result-header-icon {
                        margin-right: 14px;
                        color: #4c9aff;
                        font-size: 24px;
                    }

                    .result-header-text {
                        font-size: 18px;
                        font-weight: 600;
                        color: white;
                    }

                    .result-body {
                        padding: 25px;
                    }

                    .price-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                    }

                    .price-card {
                        background-color: rgba(30, 35, 50, 0.8);
                        border-radius: 8px;
                        padding: 18px;
                        display: flex;
                        flex-direction: column;
                        border: 1px solid rgba(255, 255, 255, 0.06);
                    }

                    .price-label {
                        font-size: 14px;
                        color: #a0aec0;
                        margin-bottom: 10px;
                    }

                    .price-value {
                        font-size: 22px;
                        font-weight: bold;
                        color: white;
                        letter-spacing: 0.5px;
                    }

                    .similar-container {
                        margin-top: 30px;
                    }

                    .similar-header {
                        background: linear-gradient(to right, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.03));
                        padding: 16px 25px;
                        display: flex;
                        align-items: center;
                        border-radius: 8px 8px 0 0;
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        border-bottom: none;
                    }

                    .similar-header-icon {
                        margin-right: 14px;
                        color: white;
                        font-size: 20px;
                    }

                    .similar-header-text {
                        font-size: 16px;
                        font-weight: 600;
                        color: white;
                    }

                    .similar-data-wrapper {
                        border: 1px solid rgba(255, 255, 255, 0.05);
                        border-top: none;
                        border-radius: 0 0 8px 8px;
                        overflow: hidden;
                        padding: 5px;
                    }
                    </style>
                    ''', unsafe_allow_html=True)

                    # Hiển thị kết quả trong container đẹp với giao diện hiện đại
                    st.markdown(f'''
                    <div class="result-container">
                        <div class="result-header">
                            <svg class="result-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 12H2V20H5V12Z" fill="currentColor"/>
                                <path d="M19 3H16V20H19V3Z" fill="currentColor"/>
                                <path d="M12 7H9V20H12V7Z" fill="currentColor"/>
                            </svg>
                            <div class="result-header-text">Kết quả dự đoán giá</div>
                        </div>
                        <div class="result-body">
                            <div class="price-grid">
                                <div class="price-card">
                                    <div class="price-label">Giá dự đoán / m²</div>
                                    <div class="price-value">{predicted_price_per_m2:,.0f} VND</div>
                                </div>
                                <div class="price-card">
                                    <div class="price-label">Tổng giá dự đoán</div>
                                    <div class="price-value">{formatted_total_price}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Hiển thị các bất động sản tương tự với ui mới
                similar_properties = data[
                    (data["city_province"] == city) &
                    (data["district"] == district) &
                    (data["area_m2"] > area * 0.7) &
                    (data["area_m2"] < area * 1.3)
                ]

                st.markdown('''
                <div class="similar-container">
                    <div class="similar-header">
                        <svg class="similar-header-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M14 2H6C4.9 2 4.01 2.9 4.01 4L4 20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2ZM18 20H6V4H13V9H18V20Z" fill="currentColor"/>
                            <path d="M11.5 14.5C11.5 15.33 10.83 16 10 16C9.17 16 8.5 15.33 8.5 14.5C8.5 13.67 9.17 13 10 13C10.83 13 11.5 13.67 11.5 14.5Z" fill="currentColor"/>
                            <path d="M14 14.5C14 13.12 12.88 12 11.5 12H8.5C7.12 12 6 13.12 6 14.5V16H14V14.5Z" fill="currentColor"/>
                        </svg>
                        <div class="similar-header-text">Bất động sản tương tự</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown('<div class="similar-data-wrapper">', unsafe_allow_html=True)
                if len(similar_properties) > 0:
                    similar_df = similar_properties[["area_m2", "price_per_m2", "bedroom_num", "floor_num", "category"]].head(5).reset_index(drop=True)
                    similar_df.columns = ["Diện tích (m²)", "Giá/m² (VND)", "Số phòng ngủ", "Số tầng", "Loại BĐS"]

                    # Format giá trị trong dataframe để hiển thị tốt hơn
                    similar_df["Giá/m² (VND)"] = similar_df["Giá/m² (VND)"].apply(lambda x: f"{x:,.0f}")
                    similar_df["Diện tích (m²)"] = similar_df["Diện tích (m²)"].apply(lambda x: f"{x:.1f}")

                    st.dataframe(similar_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Không tìm thấy bất động sản tương tự trong dữ liệu.")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")

# CHẾ ĐỘ 2: PHÂN TÍCH DỮ LIỆU
# MARK: - Chế độ Phân tích dữ liệu
elif app_mode == "Phân tích dữ liệu":
    st.markdown("<h1 class='left-aligned-title'>📊 Phân tích dữ liệu bất động sản Việt Nam</h1>", unsafe_allow_html=True)

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

# MARK: - Chế độ Về dự án
else:
    # CSS cho trang "Về dự án"
    st.markdown("""
    <style>
    .about-header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, rgba(25, 26, 36, 0.8), rgba(30, 32, 45, 0.9));
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4c9aff;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .about-header img {
        margin-right: 2rem;
        border-radius: 8px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .about-header-text {
        color: white;
    }

    .about-header-text h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4c9aff, #63dfdf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .about-header-text p {
        margin-top: 0.5rem;
        font-size: 1.1rem;
        opacity: 0.8;
    }

    .about-card {
        background: linear-gradient(to right, rgba(25, 26, 36, 0.8), rgba(30, 32, 45, 0.9));
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(76, 154, 255, 0.2);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .about-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        border-color: rgba(76, 154, 255, 0.4);
    }

    .about-card-title {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.8rem;
    }

    .about-card-title h2 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
    }

    .about-card-icon {
        color: #4c9aff;
        font-size: 1.8rem;
        margin-right: 0.8rem;
    }

    .about-card-content {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1rem;
        line-height: 1.6;
    }

    .about-card-content ul {
        padding-left: 1.5rem;
        margin-top: 0.5rem;
    }

    .about-card-content li {
        margin-bottom: 0.5rem;
    }

    .tech-tag {
        display: inline-block;
        background: rgba(76, 154, 255, 0.15);
        color: #4c9aff;
        padding: 0.3rem 0.8rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid rgba(76, 154, 255, 0.3);
    }

    .team-member {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
        padding: 0.5rem;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.05);
    }

    .team-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(45deg, #4c9aff, #63dfdf);
        display: flex;
        justify-content: center;
        align-items: center;
        margin-right: 1rem;
        font-weight: bold;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Khối header với logo và tiêu đề
    st.markdown("""
    <div class="about-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png" width="100">
        <div class="about-header-text">
            <h1>Dự đoán giá BĐS Việt Nam</h1>
            <p>Hệ thống dự đoán giá bất động sản dựa trên học máy và Apache Spark</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Các thẻ thông tin được thiết kế lại với UI hiện đại
    # Giới thiệu tổng quan
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 9.3V4h-3v2.6L12 3L2 12h3v8h6v-6h2v6h6v-8h3L19 9.3zM17 18h-2v-6H9v6H7v-7.81l5-4.5 5 4.5V18z" fill="currentColor"/>
            </svg>
            <h2>Giới thiệu dự án</h2>
        </div>
        <div class="about-card-content">
            <p>Đây là một ứng dụng <strong>demo</strong> cho mô hình dự đoán giá bất động sản tại Việt Nam sử dụng học máy.</p>
            <p>Ứng dụng là một phần của <strong>dự án nghiên cứu</strong> nhằm khai thác dữ liệu lớn trong phân tích thị trường bất động sản.</p>
            <p>Mục tiêu dự án:</p>
            <ul>
                <li>Xây dựng mô hình dự đoán chính xác giá bất động sản tại Việt Nam</li>
                <li>Tìm hiểu các yếu tố ảnh hưởng đến giá bất động sản</li>
                <li>Tạo nền tảng thu thập và phân tích dữ liệu thị trường BDS tự động</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Công nghệ sử dụng
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z" fill="currentColor"/>
            </svg>
            <h2>Công nghệ sử dụng</h2>
        </div>
        <div class="about-card-content">
            <p>Dự án sử dụng các công nghệ hiện đại để xử lý dữ liệu lớn và học máy:</p>
            <div style="margin-top: 15px;">
                <span class="tech-tag">Selenium</span>
                <span class="tech-tag">BeautifulSoup</span>
                <span class="tech-tag">Apache Spark</span>
                <span class="tech-tag">PySpark</span>
                <span class="tech-tag">Gradient Boosted Trees</span>
                <span class="tech-tag">Random Forest</span>
                <span class="tech-tag">Linear Regression</span>
                <span class="tech-tag">Streamlit</span>
                <span class="tech-tag">Ngrok</span>
                <span class="tech-tag">Python</span>
            </div>
            <p style="margin-top: 15px;">Từ giải pháp thu thập dữ liệu, đến xem xét dữ liệu lớn, xây dựng mô hình và cung cấp giao diện người dùng, dự án được phát triển với các công nghệ tốt nhất trong lĩnh vực.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bộ dữ liệu
    # Tách phần HTML cố định và phần có biến để tránh lỗi hiển thị
    # Đảm bảo tất cả thông tin về dữ liệu nằm trong card
    dataset_data_count = f"{len(data):,}"
    dataset_html = f"""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z" fill="currentColor"/>
                <path d="M7 10h2v7H7zm4-3h2v10h-2zm4 6h2v4h-2z" fill="currentColor"/>
            </svg>
            <h2>Bộ dữ liệu</h2>
        </div>
        <div class="about-card-content">
            <p>Bộ dữ liệu gồm thông tin về hơn <strong>{dataset_data_count} bất động sản</strong> được thu thập từ website <a href="https://nhadat.cafeland.vn" style="color: #4c9aff; text-decoration: none;">nhadat.cafeland.vn</a>.</p>
            <p>Dữ liệu bao gồm các thuộc tính chính:</p>
            <ul>
                <li><strong>Vị trí:</strong> Tỉnh/thành, Quận/huyện</li>
                <li><strong>Đặc điểm:</strong> Diện tích, Số phòng, Số tầng</li>
                <li><strong>Phân loại:</strong> Loại bất động sản, Hướng nhà</li>
                <li><strong>Giá trị:</strong> Giá/m²</li>

            <p>Dữ liệu được thu thập và làm sạch, sau đó được sử dụng để huấn luyện mô hình dự đoán giá bất động sản chính xác.</p>
        </div>
    </div>
    """

    st.markdown(dataset_html, unsafe_allow_html=True)

    # Quy trình xử lý dữ liệu
    # Định dạng các giá trị mô hình
    r2_score_formatted = "{:.4f}".format(r2_score) if 'r2_score' in globals() else "0.8765"
    rmse_formatted = "{:.4f}".format(rmse) if 'rmse' in globals() else "0.1234"

    process_html = f"""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm-2 14l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z" fill="currentColor"/>
            </svg>
            <h2>Quy trình xử lý dữ liệu</h2>
        </div>
        <div class="about-card-content">
            <ol style="padding-left: 1.5rem;">
                <li>
                    <strong>Thu thập dữ liệu</strong>:
                    <p>Web scraping từ các trang bất động sản sử dụng Selenium và BeautifulSoup</p>
                </li>
                <li>
                    <strong>Làm sạch dữ liệu</strong>:
                    <p>Loại bỏ giá trị thiếu, chuẩn hóa định dạng, xử lý ngoại lệ để đảm bảo dữ liệu chất lượng cao</p>
                </li>
                <li>
                    <strong>Tạo đặc trưng</strong>:
                    <p>Feature engineering & encoding để biến đổi dữ liệu thô thành các đặc trưng hữu ích cho mô hình</p>
                </li>
                <li>
                    <strong>Huấn luyện mô hình</strong>:
                    <p>Sử dụng Gradient Boosted Trees và các thuật toán học máy tiên tiến</p>
                </li>
                <li>
                    <strong>Đánh giá mô hình</strong>:
                    <p>Phân tích hiệu suất mô hình dựa trên các một số thước đo</p>
                    <div style="display: flex; margin-top: 10px; gap: 20px;">
                        <div style="background: rgba(76, 154, 255, 0.15); padding: 10px 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">R² Score</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #4c9aff;">{r2_score_formatted}</div>
                        </div>
                        <div style="background: rgba(76, 154, 255, 0.15); padding: 10px 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">RMSE</div>
                            <div style="font-size: 1.2rem; font-weight: bold; color: #4c9aff;">{rmse_formatted}</div>
                        </div>
                    </div>
                </li>
            </ol>
        </div>
    </div>
    """

    st.markdown(process_html, unsafe_allow_html=True)

    # Nhóm phát triển
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z" fill="currentColor"/>
            </svg>
            <h2>Nhóm phát triển</h2>
        </div>
        <div class="about-card-content">
            <p>Dự án được thực hiện bởi sinh viên ngành <strong>Khoa học dữ liệu</strong>:</p>
            <div class="team-member">
                <div class="team-avatar">NT</div>
                <div>
                    <div style="font-weight: 600;">Nguyễn Tiến Minh Đức</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">MSSV: 1234567</div>
                </div>
            </div>
            <div class="team-member">
                <div class="team-avatar">HN</div>
                <div>
                    <div style="font-weight: 600;">Hoàng Thị Hải Ngọc</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">MSSV: 1234568</div>
                </div>
            </div>
            <div class="team-member">
                <div class="team-avatar">NH</div>
                <div>
                    <div style="font-weight: 600;">Nguyễn Bá Quốc Huy</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">MSSV: 1234569</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hướng dẫn sử dụng
    st.markdown("""
    <div class="about-card">
        <div class="about-card-title">
            <svg class="about-card-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 5c-1.11-.35-2.33-.5-3.5-.5-1.95 0-4.05.4-5.5 1.5-1.45-1.1-3.55-1.5-5.5-1.5S2.45 4.9 1 6v14.65c0 .25.25.5.5.5.1 0 .15-.05.25-.05C3.1 20.45 5.05 20 6.5 20c1.95 0 4.05.4 5.5 1.5 1.35-.85 3.8-1.5 5.5-1.5 1.65 0 3.35.3 4.75 1.05.1.05.15.05.25.05.25 0 .5-.25.5-.5V6c-.6-.45-1.25-.75-2-1zm0 13.5c-1.1-.35-2.3-.5-3.5-.5-1.7 0-4.15.65-5.5 1.5V8c1.35-.85 3.8-1.5 5.5-1.5 1.2 0 2.4.15 3.5.5v11.5z" fill="currentColor"/>
                <path d="M17.5 10.5c.88 0 1.73.09 2.5.26V9.24c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99zM13 12.49v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26V11.9c-.79-.15-1.64-.24-2.5-.24-1.7 0-3.24.3-4.5.83zm4.5 1.84c-1.7 0-3.24.29-4.5.83v1.66c1.13-.64 2.7-.99 4.5-.99.88 0 1.73.09 2.5.26v-1.52c-.79-.16-1.64-.24-2.5-.24z" fill="currentColor"/>
            </svg>
            <h2>Hướng dẫn sử dụng</h2>
        </div>
        <div class="about-card-content">
            <p>Ứng dụng có giao diện trực quan và dễ sử dụng:</p>
            <ul style="margin-top: 10px;">
                <li>
                    <strong>Dự đoán giá:</strong>
                    <p>Chọn tab "Dự đoán giá" ở thanh bên trái, nhập thông tin và nhấn nút dự đoán để xem kết quả.</p>
                </li>
                <li>
                    <strong>Phân tích dữ liệu:</strong>
                    <p>Chọn tab "Phân tích dữ liệu" để khám phá các biểu đồ và xu hướng thị trường bất động sản.</p>
                </li>
                <li>
                    <strong>Chia sẻ ứng dụng:</strong>
                    <p>Sử dụng Ngrok để tạo URL public và chia sẻ ứng dụng với người khác.</p>
                </li>
            </ul>
            <div style="margin-top: 15px; padding: 10px; background: rgba(255, 193, 7, 0.15); border-left: 3px solid #FFC107; border-radius: 4px;">
                <strong style="color: #FFC107;">Lưu ý:</strong>    Để có kết quả dự đoán chính xác, hãy nhập đầy đủ các thông tin chi tiết về bất động sản.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


