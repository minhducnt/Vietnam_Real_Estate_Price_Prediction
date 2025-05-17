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
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }

    /* Left-aligned title styling */
    .left-aligned-title {
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
</style>
''', unsafe_allow_html=True)

# Không cần hiển thị logo riêng vì sẽ được thêm vào sidebar

# MARK: - Khởi tạo phiên Spark
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

# MARK: - Dự đoán giá
def predict_price(model, input_data):
    """Dự đoán giá dựa trên đầu vào của người dùng."""
    try:
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

        # Chuyển đổi dữ liệu sang Spark DataFrame
        spark = get_spark_session()
        spark_df = convert_to_spark(data_copy)

        # Dự đoán giá
        try:
            predictions = model.transform(spark_df)

            # Lấy kết quả dự đoán
            prediction_value = predictions.select("prediction").collect()[0][0]

            return prediction_value
        except Exception as transform_error:
            st.error(f"Lỗi khi chuyển đổi dữ liệu: {transform_error}")
            # Hiển thị thông tin bổ sung về mô hình để debug
            st.write("Thông tin về mô hình:")
            st.write(str(model)[:500] + "..." if len(str(model)) > 500 else str(model))
            return None
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

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
    /* Cải thiện style cho button */
    button[kind="primary"], button[kind="secondary"] {
        padding-left: 16px !important;
        padding-right: 6px !important;
    }

    /* Cải thiện cách hiển thị icon trong button */
    button[kind="primary"] p:first-child, button[kind="secondary"] p:first-child {
        display: inline-block;
        margin-right: 8px !important;
        font-size: 1.2rem !important;
    }

    /* Tăng kích thước chữ cho nút */
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

# Sử dụng columns để hiển thị metrics độ chính xác và RMSE
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown("""
    <div class="enhanced-metric-card">
        <div class="metric-title" style="text-align:center;">RẤ Score</div>
        <div class="metric-value">{r2_score:.4f}</div>
        <div class="metric-description">Độ chính xác</div>
    </div>
    """.format(r2_score=r2_score), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="enhanced-metric-card">
        <div class="metric-title" style="text-align:center;">RMSE</div>
        <div class="metric-value">{rmse:.4f}</div>
        <div class="metric-description">Số liệu chính xác</div>
    </div>
    """.format(rmse=rmse), unsafe_allow_html=True)

# Thêm khoảng cách giữa các card metric và số lượng dữ liệu
st.sidebar.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Số lượng bất động sản - hiển thị với card rộng hơn
st.sidebar.markdown("""
<div class="enhanced-metric-card" style="background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7)); border-color: rgba(76,255,154,0.3); height: 125px; margin-top: 10px;">
    <div class="metric-header">
        <div class="metric-icon" style="background-color: rgba(76,255,154,0.2);">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 3V21H21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M19 5L9 15L6 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">Số lượng dữ liệu</span>
    </div>
    <div class="metric-value" style="color: #4dff9e; font-size: 1.8rem;">{data_count:,}</div>
    <div class="metric-description">Bất động sản trong dữ liệu</div>
</div>
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
    st.markdown("<h1 class='left-aligned-title'>🏘️ Dự đoán giá bất động sản Việt Nam</h1>", unsafe_allow_html=True)
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
    st.markdown("<h1 class='left-aligned-title'>ℹ️ Về dự án dự đoán giá bất động sản Việt Nam</h1>", unsafe_allow_html=True)

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

