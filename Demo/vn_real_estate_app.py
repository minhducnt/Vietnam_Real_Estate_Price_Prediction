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

# Load CSS từ file riêng biệt để tạo giao diện hiện đại
def load_css(css_file):
    try:
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
        st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
        print(f"Loaded CSS successfully from {css_file}")
        return True
    except Exception as e:
        print(f"Error loading CSS: {e}")
        return False

# Load CSS từ file riêng biệt
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'styles', 'main.css')
if not load_css(css_path):
    st.error(f"Failed to load CSS from {css_path}. UI may not display correctly.")
    st.markdown("""
    <style>
    .sidebar-header {background: linear-gradient(to right, #2c5282, #1a365d); padding: 1.5rem 1rem; text-align: center; margin-bottom: 1.6rem; border-bottom: 1px solid rgba(255,255,255,0.1); border-radius: 0.8rem;}
    .sidebar-header h2 {color: white; margin: 0; font-size: 1.3rem;}
    .sidebar-header p {color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;}
    .sidebar-header img {max-width: 40px; margin-bottom: 0.5rem;}
    .enhanced-metric-card {border-radius: 10px; padding: 0.75rem; margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;}
    .blue-gradient {background: linear-gradient(145deg, rgba(51,97,255,0.3), rgba(29,55,147,0.5)); border-color: rgba(100,149,237,0.3);}
    .purple-gradient {background: linear-gradient(145deg, rgba(139,92,246,0.3), rgba(76,29,149,0.5)); border-color: rgba(167,139,250,0.3);}
    .green-gradient {background: linear-gradient(145deg, rgba(44,130,96,0.5), rgba(26,93,59,0.7)); border-color: rgba(76,255,154,0.3);}
    </style>
    """, unsafe_allow_html=True)



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

# MARK: - Chuyển đổi Spark
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

    # Chuyển đổi sang Spark
    data_spark = convert_to_spark(data)

    # Định nghĩa các cột để sử dụng trong mô hình
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

# MARK: - Main App Flow
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
st.sidebar.markdown("""
<div class="sidebar-header">
    <img src="https://img.icons8.com/fluency/96/000000/home.png" alt="Logo">
    <h2>BĐS Việt Nam</h2>
    <p>AI Dự Đoán Giá</p>
    <p>Nhóm 05</p>
</div>
""", unsafe_allow_html=True)

# Set session state for app_mode if it doesn't exist
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = "Dự đoán giá"

# Phương thức để cập nhật app_mode
def set_app_mode(mode):
    st.session_state['app_mode'] = mode

# Lấy mode hiện tại
app_mode = st.session_state['app_mode']

# Danh sách các chế độ ứng dụng
app_modes = ["Dự đoán giá", "Phân tích dữ liệu", "Thống kê", "Về dự án"]

# Container cho menu
menu_container = st.sidebar.container()

# Tạo các button
for i, mode in enumerate(app_modes):
    if menu_container.button(mode, key=f"nav_{i}",
                           use_container_width=True,
                           on_click=set_app_mode,
                           args=(mode,),
                           type="primary" if mode == app_mode else "secondary"):
        pass

# Hiển thị thông tin mô hình trong nhóm
st.sidebar.markdown('<div class="model-stats-container"><div class="metric-header"><div class="metric-icon"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 16V8.00002C20.9996 7.6493 20.9071 7.30483 20.7315 7.00119C20.556 6.69754 20.3037 6.44539 20 6.27002L13 2.27002C12.696 2.09449 12.3511 2.00208 12 2.00208C11.6489 2.00208 11.304 2.09449 11 2.27002L4 6.27002C3.69626 6.44539 3.44398 6.69754 3.26846 7.00119C3.09294 7.30483 3.00036 7.6493 3 8.00002V16C3.00036 16.3508 3.09294 16.6952 3.26846 16.9989C3.44398 17.3025 3.69626 17.5547 4 17.73L11 21.73C11.304 21.9056 11.6489 21.998 12 21.998C12.3511 21.998 12.696 21.9056 13 21.73L20 17.73C20.3037 17.5547 20.556 17.3025 20.7315 16.9989C20.9071 16.6952 20.9996 16.3508 21 16Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><span class="metric-title">Thông số mô hình</span></div>', unsafe_allow_html=True)

# Metrics độ chính xác
st.sidebar.markdown("""
<div class="enhanced-metric-card blue-gradient">
    <div class="metric-header">
        <div class="metric-icon blue-icon-bg">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">R² Score</span>
    </div>
    <div class="clean-metric-value blue-value">{r2_score:.4f}</div>
</div>
""".format(r2_score=r2_score), unsafe_allow_html=True)

# Thêm khoảng cách giữa hai card thông số mô hình
st.sidebar.markdown("""<div class="spacer-20"></div>""", unsafe_allow_html=True)

# Metrics độ lệch chuẩn - RMSE
st.sidebar.markdown("""
<div class="enhanced-metric-card purple-gradient">
    <div class="metric-header">
        <div class="metric-icon purple-icon-bg">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9 22V12H15V22" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <span class="metric-title">RMSE</span>
    </div>
    <div class="clean-metric-value purple-value">{rmse:.4f}</div>
</div>
""".format(rmse=rmse), unsafe_allow_html=True)

# Footer của sidebar
st.sidebar.markdown("""<hr class="hr-divider">""", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
        <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 16V12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 8H12.01" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>Dự đoán giá BĐS Việt Nam</span>
</div>

<div class="flex-container">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="info-icon">
        <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 7.61305 3.94821 5.32387 5.63604 3.63604C7.32387 1.94821 9.61305 1 12 1C14.3869 1 16.6761 1.94821 18.364 3.63604C20.0518 5.32387 21 7.61305 21 10Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M12 13C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span>Nguồn: nhadat.cafeland.vn</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# MARK: - Chế độ Dự đoán giá
if app_mode == "Dự đoán giá":
    # Tiêu đề trang
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

    # Các nút được định dạng từ file CSS riêng
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
    # Tiêu đề trang
    statistics_header = """
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                    <line x1="2" y1="20" x2="22" y2="20"></line>
                </svg>
            </div>
            <div class="header-text">Phân tích dữ liệu</div>
        </div>
        <div class="header-desc">
            Phân tích dữ liệu bất động sản tại Việt Nam
        </div>
    </div>
    """
    st.markdown(statistics_header, unsafe_allow_html=True)

    # Tạo tabs để phân chia nội dung
    tab1, tab2, tab3 = st.tabs(["Phân phối giá", "Phân tích vị trí", "Đặc điểm bất động sản"])

    with tab1:
        st.markdown("## Phân tích phân phối giá bất động sản")

        # Thông tin thống kê tổng quan
        avg_price = data["price_per_m2"].mean()
        max_price = data["price_per_m2"].max()
        min_price = data["price_per_m2"].min()
        median_price = data["price_per_m2"].median()

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung bình/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung vị/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá cao nhất/m²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số BĐS</div>
            </div>
        </div>
        """.format(avg_price, median_price, max_price, len(data)), unsafe_allow_html=True)

        # Card 1: Biểu đồ phân phối giá
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 3v18h18"></path>
                        <path d="M18 12h-2"></path>
                        <path d="M13 8h-2"></path>
                        <path d="M8 16H6"></path>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Phân phối giá bất động sản</div>
                    <div class="chart-desc">So sánh phân phối giá gốc và phân phối giá sau biến đổi logarit</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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

        # Thêm khoảng trống
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 2: Lọc theo khoảng giá
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Lọc dữ liệu theo khoảng giá</div>
                    <div class="chart-desc">Tìm kiếm bất động sản trong khoảng giá mong muốn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        price_range = st.slider(
            "Chọn khoảng giá (VND/m²)",
            min_value=float(data["price_per_m2"].min()),
            max_value=float(data["price_per_m2"].max()),
            value=(float(data["price_per_m2"].quantile(0.25)), float(data["price_per_m2"].quantile(0.75)))
        )

        # Thêm khoảng trống sau slider
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)

        # Lọc dữ liệu theo khoảng giá đã chọn
        filtered_data = data[(data["price_per_m2"] >= price_range[0]) & (data["price_per_m2"] <= price_range[1])]

        # Tính toán phần trăm
        total_count = len(data)
        filtered_count = len(filtered_data)
        percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0

        # Thêm khoảng trống trước thông báo tìm kiếm
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Hiển thị kết quả tìm kiếm
        st.markdown(f"""
        <div style="display: flex; align-items: center; background-color: #1E293B; border-radius: 12px; padding: 15px; margin-bottom: 25px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-left: 4px solid #4C9AFF;">
            <div style="background-color: rgba(76, 154, 255, 0.15); width: 42px; height: 42px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 16px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4C9AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
            </div>
            <div>
                <div style="font-size: 16px; font-weight: 500; color: #E2E8F0; margin-bottom: 5px;">
                    Đã tìm thấy <span style="font-weight: 700; color: #4C9AFF;">{filtered_count:,}</span> bất động sản
                </div>
                <div style="font-size: 13px; color: #94A3B8;">
                    Trong khoảng giá đã chọn • Chiếm <span style="font-weight: 600; color: #A5B4FC;">{int(percentage)}%</span> tổng số dữ liệu
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Thêm khoảng trống sau thông báo
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Hiển thị thông tin về dữ liệu đã lọc trong một dòng
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{len(filtered_data):,}</div>
                <div class="stat-label">Số lượng BĐS</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['price_per_m2'].mean():,.0f}</div>
                <div class="stat-label">Giá trung bình/m² (VND)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card" style="margin: 0 0 30px 0; width: 100%;">
                <div class="stat-value">{filtered_data['area_m2'].mean():.1f}</div>
                <div class="stat-label">Diện tích trung bình (m²)</div>
            </div>
            """, unsafe_allow_html=True)

        # Hiển thị dữ liệu đã lọc với card
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="8" y1="6" x2="21" y2="6"></line>
                        <line x1="8" y1="12" x2="21" y2="12"></line>
                        <line x1="8" y1="18" x2="21" y2="18"></line>
                        <line x1="3" y1="6" x2="3.01" y2="6"></line>
                        <line x1="3" y1="12" x2="3.01" y2="12"></line>
                        <line x1="3" y1="18" x2="3.01" y2="18"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Dữ liệu bất động sản đã lọc</div>
                    <div class="chart-desc">Danh sách 10 bất động sản đầu tiên trong khoảng giá đã chọn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(filtered_data[["city_province", "district", "area_m2", "price_per_m2", "category"]].head(10), use_container_width=True)

    with tab2:
        st.markdown("## Phân tích giá theo vị trí địa lý")

        # Tổng hợp thông tin theo khu vực
        total_provinces = data["city_province"].nunique()
        total_districts = data["district"].nunique()
        top_province = data["city_province"].value_counts().index[0]
        top_district = data["district"].value_counts().index[0]

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số tỉnh/TP</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số quận/huyện</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Khu vực phổ biến nhất</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Quận/huyện phổ biến nhất</div>
            </div>
        </div>
        """.format(total_provinces, total_districts, top_province, top_district), unsafe_allow_html=True)

        # Card 1: Giá trung bình theo tỉnh/thành phố
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="7" width="3" height="10"></rect>
                        <rect x="8" y="5" width="3" height="12"></rect>
                        <rect x="14" y="3" width="3" height="14"></rect>
                        <rect x="20" y="9" width="3" height="8"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo tỉnh/thành phố</div>
                    <div class="chart-desc">Top 10 tỉnh/thành phố có giá bất động sản cao nhất</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tính giá trung bình theo tỉnh/thành phố
        city_price = data.groupby("city_province")["price_per_m2"].mean().sort_values(ascending=False).reset_index()
        city_price.columns = ["Tỉnh/Thành phố", "Giá trung bình/m²"]

        # Vẽ biểu đồ
        fig = px.bar(
            city_price.head(10),
            x="Tỉnh/Thành phố",
            y="Giá trung bình/m²",
            color="Giá trung bình/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='#333333'))
        )
        st.plotly_chart(fig, use_container_width=True)

        # Thêm khoảng trống
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 2: Giá trung bình theo quận/huyện
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                        <circle cx="12" cy="12" r="4"></circle>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo quận/huyện</div>
                    <div class="chart-desc">Phân tích chi tiết theo khu vực đã chọn</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            color="Giá trung bình/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='#333333'))
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("## Phân tích mối quan hệ giữa đặc điểm và giá")

        # Thống kê tổng quan về đặc điểm bất động sản
        avg_area = data["area_m2"].mean()
        avg_bedroom = data["bedroom_num"].mean()
        price_area_corr = data[["price_per_m2", "area_m2"]].corr().iloc[0, 1]
        numeric_features_count = len([col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])])

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:.1f}</div>
                <div class="stat-label">Diện tích trung bình (m²)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.1f}</div>
                <div class="stat-label">Số phòng ngủ trung bình</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:.2f}</div>
                <div class="stat-label">Tương quan giá-diện tích</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:d}</div>
                <div class="stat-label">Số đặc trưng số</div>
            </div>
        </div>
        """.format(avg_area, avg_bedroom, price_area_corr, numeric_features_count), unsafe_allow_html=True)

        # Card 1: Tương quan giữa diện tích và giá
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="6" cy="6" r="3"></circle>
                        <circle cx="18" cy="18" r="3"></circle>
                        <line x1="9" y1="9" x2="15" y2="15"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Mối quan hệ giữa diện tích và giá</div>
                    <div class="chart-desc">Phân tích sự tương quan giữa diện tích và giá theo khu vực và số phòng ngủ</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tạo mẫu nhỏ hơn nếu có quá nhiều dữ liệu
        sample_size = min(1000, len(data))
        sampled_data = data.sample(n=sample_size, random_state=42)

        # Tạo bản sao và thêm cột size_value để đảm bảo giá trị không âm cho thuộc tính size
        plot_data = sampled_data.copy()
        # Chuyển đổi giá trị âm thành 1 và đảm bảo tất cả các giá trị đều > 0
        plot_data['size_value'] = plot_data['bedroom_num'].apply(lambda x: max(1, x) if pd.notna(x) else 1)

        # Lọc dữ liệu trong khoảng hợp lý để biểu đồ đẹp hơn
        filtered_data = plot_data[
            (plot_data["price_per_m2"] < plot_data["price_per_m2"].quantile(0.99)) &
            (plot_data["area_m2"] < plot_data["area_m2"].quantile(0.99))
        ]

        # Vẽ biểu đồ phân tán với size_value
        fig = px.scatter(
            filtered_data,
            x="area_m2",
            y="price_per_m2",
            color="city_province",
            size="size_value",  # Sử dụng cột size_value mới thay vì bedroom_num
            hover_data=["district", "category", "bedroom_num"],  # Vẫn hiển thị bedroom_num trong hover
            labels={
                "area_m2": "Diện tích (m²)",
                "price_per_m2": "Giá/m² (VND)",
                "city_province": "Tỉnh/Thành phố",
                "bedroom_num": "Số phòng ngủ"
            }
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='#333333'))
        )
        st.plotly_chart(fig, use_container_width=True)

        # Thêm khoảng trống
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 2: Ma trận tương quan
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <line x1="3" y1="9" x2="21" y2="9"></line>
                        <line x1="3" y1="15" x2="21" y2="15"></line>
                        <line x1="9" y1="3" x2="9" y2="21"></line>
                        <line x1="15" y1="3" x2="15" y2="21"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Ma trận tương quan giữa các đặc điểm</div>
                    <div class="chart-desc">Phân tích mối tương quan giữa các đặc trưng số trong dữ liệu</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chọn các đặc trưng số để tính tương quan
        numeric_features = ["area_m2", "bedroom_num", "floor_num", "toilet_num", "livingroom_num", "street_width_m", "price_per_m2"]
        corr_matrix = data[numeric_features].corr()

        # Vẽ heatmap tương quan
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        plt.title("Ma trận tương quan giữa các đặc điểm")
        st.pyplot(fig)

        # Thêm khoảng trống
        st.markdown('<div style="height: 30px;"></div>', unsafe_allow_html=True)

        # Card 3: Phân tích theo đặc điểm
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                        <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                        <line x1="12" y1="22.08" x2="12" y2="12"></line>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Phân tích giá theo đặc điểm</div>
                    <div class="chart-desc">So sánh giá trung bình theo các đặc điểm khác nhau của BĐS</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
            color="Giá trung bình/m²",
            color_continuous_scale='Viridis',
            template="plotly_white"
        )

        # Cập nhật layout của biểu đồ
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='#333333'))
        )
        st.plotly_chart(fig, use_container_width=True)

# MARK: - Chế độ Thống kê
elif app_mode == "Thống kê":
    # Tiêu đề trang
    statistics_header = """
    <div class="modern-header">
        <div class="header-title">
            <div class="header-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6" y1="20" x2="6" y2="14"></line>
                    <line x1="2" y1="20" x2="22" y2="20"></line>
                </svg>
            </div>
            <div class="header-text">Thống kê dữ liệu</div>
        </div>
        <div class="header-desc">
            Thông tin thống kê về dữ liệu bất động sản tại Việt Nam
        </div>
    </div>
    """
    st.markdown(statistics_header, unsafe_allow_html=True)

    # Kiểm tra tên cột liên quan đến tỉnh/thành phố trong dữ liệu
    location_column = None
    possible_location_columns = ['province', 'city', 'location', 'tinh_thanh', 'thanh_pho', 'district']

    for col in possible_location_columns:
        if col in data.columns:
            location_column = col
            break

    if location_column is None:
        st.error("Không tìm thấy cột chứa thông tin về tỉnh/thành phố trong dữ liệu. Vui lòng kiểm tra lại dữ liệu.")
        # Nếu không tìm thấy, sử dụng cột đầu tiên làm ví dụ
        if len(data.columns) > 0:
            location_column = data.columns[0]
            st.warning(f"Đang sử dụng cột '{location_column}' làm dữ liệu thay thế cho mục đích hiển thị.")
        else:
            st.stop()

    # Kiểm tra tên cột liên quan đến giá trong dữ liệu
    price_column = None
    possible_price_columns = ['price', 'gia', 'price_per_m2', 'price_m2', 'gia_tien', 'total_price', 'value']

    for col in possible_price_columns:
        if col in data.columns:
            price_column = col
            break

    if price_column is None:
        st.error("Không tìm thấy cột chứa thông tin về giá trong dữ liệu. Vui lòng kiểm tra lại dữ liệu.")
        # Tìm cột chứa dữ liệu số để sử dụng làm giá
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                price_column = col
                st.warning(f"Đang sử dụng cột số '{price_column}' làm dữ liệu thay thế cho giá.")
                break

        if price_column is None and len(data.columns) > 0:
            price_column = data.columns[0]  # Sử dụng cột đầu tiên nếu không có cột số
            st.warning(f"Đang sử dụng cột '{price_column}' làm dữ liệu thay thế cho giá.")

        if price_column is None:
            st.stop()

    # Thêm một container chứa các tab
    tab1, tab2, tab3 = st.tabs(["Phân phối giá", "Phân tich khu vực", "Tương quan"])

    # Tab 1: Phân phối giá
    with tab1:
        st.markdown("## Phân phối giá bất động sản")

        # Thông tin thống kê tổng quan
        avg_price = data[price_column].mean()
        max_price = data[price_column].max()
        min_price = data[price_column].min()
        median_price = data[price_column].median()

        # Hiển thị thống kê tổng quan trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung bình (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá trung vị (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Giá cao nhất (triệu VNĐ)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số BĐS</div>
            </div>
        </div>
        """.format(avg_price, median_price, max_price, len(data)), unsafe_allow_html=True)

        # Card 1: Phân tích giá trung bình theo thành phố/tỉnh
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="2" y="7" width="3" height="10"></rect>
                        <rect x="8" y="5" width="3" height="12"></rect>
                        <rect x="14" y="3" width="3" height="14"></rect>
                        <rect x="20" y="9" width="3" height="8"></rect>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Giá trung bình theo khu vực</div>
                    <div class="chart-desc">Top 10 khu vực có giá BĐS trung bình cao nhất</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tính giá trung bình theo tỉnh/thành phố
        avg_price_by_location = data.groupby(location_column)[price_column].mean().reset_index()
        avg_price_by_location = avg_price_by_location.sort_values(price_column, ascending=False)

        # Vẽ biểu đồ cột cho giá trung bình
        fig1 = px.bar(avg_price_by_location.head(10), x=location_column, y=price_column,
                    labels={price_column:'Giá trung bình (triệu VNĐ)', location_column:location_column.replace('_', ' ').title()},
                    color=price_column,
                    color_continuous_scale='Viridis')

        # Cập nhật layout của biểu đồ
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='white'))
        )
        fig1.update_xaxes(tickfont=dict(color='white'))
        fig1.update_yaxes(tickfont=dict(color='white'))

        with st.container():
            st.plotly_chart(fig1, use_container_width=True)

        # Card 2: Biểu đồ phân phối giá
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M3 3v18h18"></path>
                        <path d="M18 12h-2"></path>
                        <path d="M13 8h-2"></path>
                        <path d="M8 16H6"></path>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Phân phối giá bất động sản</div>
                    <div class="chart-desc">Số lượng BĐS theo các khoảng giá khác nhau (không bao gồm 1% BĐS có giá cao nhất)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Lọc dữ liệu giá trong khoảng hợp lý để biểu đồ đẹp hơn
        filtered_data = data[data[price_column] < data[price_column].quantile(0.99)]

        # Vẽ biểu đồ histogram
        fig2 = px.histogram(filtered_data, x=price_column, nbins=50,
                        labels={price_column:'Giá (triệu VNĐ)'},
                        opacity=0.8,
                        color_discrete_sequence=['#4169E1'])

        # Cập nhật layout của biểu đồ
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=0, b=0, l=0, r=0)
        )
        fig2.update_xaxes(tickfont=dict(color='white'))
        fig2.update_yaxes(tickfont=dict(color='white'))

        with st.container():
            st.plotly_chart(fig2, use_container_width=True)

    # Tab 2: Phân tích khu vực
    with tab2:
        st.markdown("## Phân tích theo khu vực")

        # Tổng hợp thông tin theo khu vực
        total_regions = data[location_column].nunique()
        top_region = data[location_column].value_counts().index[0]
        region_max_count = data[location_column].value_counts().max()
        region_min_count = data[location_column].value_counts().min()

        # Hiển thị thống kê trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Tổng số khu vực</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Khu vực phổ biến nhất</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Số BĐS nhiều nhất</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Số BĐS ít nhất</div>
            </div>
        </div>
        """.format(total_regions, top_region, region_max_count, region_min_count), unsafe_allow_html=True)

        # Card 1: Số lượng bất động sản theo khu vực
        st.markdown("""
        <div class="chart-card">
            <div class="chart-header">
                <div class="chart-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
                        <circle cx="12" cy="12" r="4"></circle>
                    </svg>
                </div>
                <div class="chart-title-container">
                    <div class="chart-title">Số lượng BĐS theo khu vực</div>
                    <div class="chart-desc">Top 10 khu vực có nhiều bất động sản nhất trong dữ liệu</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        property_count_by_location = data[location_column].value_counts().reset_index()
        property_count_by_location.columns = [location_column, 'count']
        property_count_by_location = property_count_by_location.sort_values('count', ascending=False)

        # Vẽ biểu đồ cột cho số lượng BDS
        fig3 = px.bar(property_count_by_location.head(10), x=location_column, y='count',
                    labels={'count':'Số lượng bất động sản', location_column:location_column.replace('_', ' ').title()},
                    color='count',
                    color_continuous_scale='Viridis')

        # Cập nhật layout của biểu đồ
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=0, b=0, l=0, r=0),
            coloraxis_colorbar=dict(tickfont=dict(color='white'))
        )
        fig3.update_xaxes(tickfont=dict(color='white'))
        fig3.update_yaxes(tickfont=dict(color='white'))

        with st.container():
            st.plotly_chart(fig3, use_container_width=True)

        # Kiểm tra cột diện tích
        area_column = 'area' if 'area' in data.columns else None
        if area_column is None:
            for col in ['dien_tich', 'area_m2', 'square_meters', 'dientich', 'dt']:
                if col in data.columns:
                    area_column = col
                    break

        if area_column is None:
            # Tìm cột số khác nếu không có area
            for col in data.columns:
                if col != price_column and pd.api.types.is_numeric_dtype(data[col]):
                    area_column = col
                    st.warning(f"Đang sử dụng cột '{area_column}' làm dữ liệu thay thế cho diện tích.")
                    break

        if area_column is None:
            st.error("Không tìm thấy cột chứa thông tin về diện tích trong dữ liệu.")
        else:
            # Card 2: Phân phối diện tích theo tỉnh/thành phố
            st.markdown("""
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                    </div>
                    <div class="chart-title-container">
                        <div class="chart-title">Diện tích trung bình theo khu vực</div>
                        <div class="chart-desc">Top 10 khu vực có diện tích trung bình lớn nhất</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            avg_area_by_location = data.groupby(location_column)[area_column].mean().reset_index()
            avg_area_by_location = avg_area_by_location.sort_values(area_column, ascending=False)

            fig4 = px.bar(avg_area_by_location.head(10), x=location_column, y=area_column,
                        labels={area_column:'Diện tích trung bình (m²)', location_column:location_column.replace('_', ' ').title()},
                        color=area_column,
                        color_continuous_scale='Viridis')

            # Cập nhật layout của biểu đồ
            fig4.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=0, b=0, l=0, r=0),
                coloraxis_colorbar=dict(tickfont=dict(color='white'))
            )
            fig4.update_xaxes(tickfont=dict(color='white'))
            fig4.update_yaxes(tickfont=dict(color='white'))

            with st.container():
                st.plotly_chart(fig4, use_container_width=True)

    # Tab 3: Tương quan
    with tab3:
        st.markdown("## Phân tích tương quan các yếu tố")

        # Tổng hợp thông tin về các yếu tố tương quan
        corr_variables = 0
        price_range = 0

        # Đếm số cột số có thể dùng trong tương quan
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                corr_variables += 1

        # Tính giá cao nhất và thấp nhất (không tính ngoại lệ)
        if price_column in data.columns:
            price_range = data[data[price_column] < data[price_column].quantile(0.99)][price_column].max() - \
                          data[data[price_column] > data[price_column].quantile(0.01)][price_column].min()

        # Hiển thị thống kê trong grid
        st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="data-grid">
            <div class="stat-card">
                <div class="stat-value">{:,d}</div>
                <div class="stat-label">Số biến số liên tục</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{:,.0f}</div>
                <div class="stat-label">Khoảng giá (triệu VNĐ)</div>
            </div>
        </div>
        """.format(corr_variables, price_range), unsafe_allow_html=True)

        # Kiểm tra cột diện tích
        area_column = 'area' if 'area' in data.columns else None
        if area_column is None:
            for col in ['dien_tich', 'area_m2', 'square_meters', 'dientich']:
                if col in data.columns:
                    area_column = col
                    break

        if area_column is None:
            # Tìm cột số khác nếu không có area
            for col in data.columns:
                if col != price_column and pd.api.types.is_numeric_dtype(data[col]):
                    area_column = col
                    st.warning(f"Đang sử dụng cột '{area_column}' làm dữ liệu thay thế cho diện tích.")
                    break

        if area_column is None and len(data.columns) > 1:
            # Sử dụng cột đầu tiên khác với giá
            for col in data.columns:
                if col != price_column:
                    area_column = col
                    st.warning(f"Đang sử dụng cột '{area_column}' làm dữ liệu thay thế cho diện tích.")
                    break

        if area_column is None:
            st.error("Không đủ dữ liệu để vẽ biểu đồ tương quan.")
        else:
            # Lọc dữ liệu trong khoảng hợp lý để biểu đồ đẹp hơn
            filtered_data = data[data[price_column] < data[price_column].quantile(0.99)]
            filtered_data = filtered_data[filtered_data[area_column] < filtered_data[area_column].quantile(0.99)]

            # Card 1: Tương quan giữa diện tích và giá
            st.markdown("""
            <div class="chart-card">
                <div class="chart-header">
                    <div class="chart-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="6" cy="6" r="3"></circle>
                            <circle cx="18" cy="18" r="3"></circle>
                            <line x1="9" y1="9" x2="15" y2="15"></line>
                        </svg>
                    </div>
                    <div class="chart-title-container">
                        <div class="chart-title">Tương quan giữa diện tích và giá</div>
                        <div class="chart-desc">Phân tích mối quan hệ giữa diện tích và giá của BĐS theo khu vực</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Vẽ biểu đồ scatter plot
            fig5 = px.scatter(filtered_data, x=area_column, y=price_column,
                          labels={area_column:'Diện tích (m²)', price_column:'Giá (triệu VNĐ)'},
                          opacity=0.6,
                          color=location_column if location_column in filtered_data.columns else None,
                          size_max=10,
                          height=500)

            # Cập nhật layout của biểu đồ
            fig5.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=0, b=0, l=0, r=0),
                legend=dict(font=dict(color='white')),
            )
            fig5.update_xaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')
            fig5.update_yaxes(tickfont=dict(color='white'), gridcolor='rgba(255,255,255,0.1)')

            with st.container():
                st.plotly_chart(fig5, use_container_width=True)

            # Tương quan giữa các biến số
            # Chọn các biến số liên tục để tính tương quan
            numerical_columns = []
            possible_numerical_features = [
                price_column, area_column,
                'bedrooms', 'bathrooms', 'floor', 'floors',
                'rooms', 'so_phong', 'so_tang', 'phong_ngu', 'phong_tam'
            ]

            # Thêm một danh sách các cột số đã biết
            known_numerical_columns = set(possible_numerical_features)

            # Tìm tất cả các cột số trong dữ liệu
            for col in filtered_data.columns:
                if col in known_numerical_columns or pd.api.types.is_numeric_dtype(filtered_data[col]):
                    if col not in numerical_columns:  # Tránh trùng lắp
                        numerical_columns.append(col)

            if len(numerical_columns) >= 2:
                # Lấy tối đa 5 cột số để ma trận tương quan đơn giản
                numerical_columns = numerical_columns[:5]

                try:
                    corr_data = filtered_data[numerical_columns].corr()

                    # Card 2: Ma trận tương quan
                    st.markdown("""
                    <div class="chart-card">
                        <div class="chart-header">
                            <div class="chart-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <rect x="3" y="3" width="7" height="7"></rect>
                                    <rect x="14" y="3" width="7" height="7"></rect>
                                    <rect x="14" y="14" width="7" height="7"></rect>
                                    <rect x="3" y="14" width="7" height="7"></rect>
                                </svg>
                            </div>
                            <div class="chart-title-container">
                                <div class="chart-title">Ma trận tương quan giữa các yếu tố</div>
                                <div class="chart-desc">Phân tích mức độ tương quan giữa các yếu tố chính</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Vẽ ma trận tương quan
                    fig6 = px.imshow(corr_data,
                               text_auto=True,
                               color_continuous_scale='RdBu_r')

                    # Cập nhật layout của biểu đồ
                    fig6.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(t=0, b=0, l=0, r=0),
                        coloraxis_colorbar=dict(tickfont=dict(color='white'))
                    )
                    fig6.update_xaxes(tickfont=dict(color='white'))
                    fig6.update_yaxes(tickfont=dict(color='white'))

                    with st.container():
                        st.plotly_chart(fig6, use_container_width=True)
                except Exception as e:
                    st.error(f"Không thể tạo ma trận tương quan: {str(e)}")
            else:
                st.warning("Không đủ dữ liệu số để tạo ma trận tương quan.")

# MARK: - Chế độ Về dự án
elif app_mode == "Về dự án":
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
        </div>
        <p>Dữ liệu được thu thập và làm sạch, sau đó được sử dụng để huấn luyện mô hình dự đoán giá bất động sản chính xác.</p>
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


