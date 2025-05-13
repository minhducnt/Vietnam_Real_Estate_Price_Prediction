#!/bin/bash

# Script khởi chạy ứng dụng demo Dự đoán giá bất động sản Việt Nam
# với Streamlit và Ngrok
# Tác giả: Cascade AI
# Ngày: 13/05/2025

echo "===== KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Kiểm tra môi trường
if ! command -v python3 &> /dev/null; then
    echo "❌ Không tìm thấy Python 3. Vui lòng cài đặt Python 3 trước khi chạy ứng dụng."
    exit 1
fi

# Tạo và kích hoạt môi trường ảo nếu cần
if [ ! -d "venv" ]; then
    echo "🔧 Tạo môi trường ảo mới..."
    python3 -m venv venv
    CREATED_VENV=1
else
    CREATED_VENV=0
fi

echo "🚀 Kích hoạt môi trường ảo..."
source venv/bin/activate

# Cài đặt các thư viện nếu vừa tạo môi trường mới
if [ $CREATED_VENV -eq 1 ]; then
    echo "📦 Cài đặt các thư viện cần thiết..."
    pip install --upgrade pip setuptools wheel
    pip install streamlit pandas numpy matplotlib seaborn plotly pyspark folium pyngrok ngrok
fi

# Chuyển đến thư mục Core và chạy ứng dụng
cd "$(dirname "$0")"

echo "🌐 Bạn muốn chạy ứng dụng với ngrok để tạo URL public không? (y/n)"
read use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    echo "🔑 Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
    read -s ngrok_token
    
    echo "⚙️ Cấu hình ngrok và khởi chạy Streamlit..."
    
    # Tạo file Python tạm thời để chạy ngrok
    cat > run_with_ngrok.py << EOF
import os
import subprocess
import time
from pyngrok import ngrok

# Thiết lập ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con
streamlit_process = subprocess.Popen(["streamlit", "run", "vn_real_estate_app.py"])

# Tạo tunnel HTTP đến cổng Streamlit
http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print("\n" + "="*60)
print(f"🌐 URL NGROK PUBLIC: {http_tunnel.public_url}")
print("🔗 Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
print("="*60 + "\n")

try:
    # Giữ script chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Dọn dẹp khi người dùng nhấn Ctrl+C
    print("\n🛑 Đang dừng ứng dụng...")
    ngrok.kill()
    streamlit_process.terminate()
EOF

    # Chạy script Python với ngrok
    python run_with_ngrok.py
    
    # Xóa file tạm thời sau khi chạy
    rm run_with_ngrok.py
    
else
    echo "🖥️ Khởi chạy Streamlit trên localhost:8501..."
    streamlit run vn_real_estate_app.py
fi

# Trở về thư mục gốc và deactivate môi trường ảo khi kết thúc
cd - > /dev/null
deactivate
