#!/bin/bash

# Script khởi chạy ứng dụng demo Dự đoán giá bất động sản Việt Nam
# với Streamlit và Ngrok

echo "===== KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Kiểm tra thư mục App
if [ ! -d "App" ]; then
    echo "Không tìm thấy thư mục App. Vui lòng kiểm tra lại cấu trúc dự án!"
    exit 1
fi

# Kiểm tra file requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "⚠️ Cảnh báo: Không tìm thấy file requirements.txt. Sẽ sử dụng danh sách thư viện mặc định."
fi

# Cài đặt python-setuptools nếu chưa có
if ! brew list python-setuptools &>/dev/null; then
    echo "Cài đặt python-setuptools..."
    brew install python-setuptools
fi

# Kiểm tra và cài đặt các thư viện hệ thống cần thiết
echo "Kiểm tra các thư viện hệ thống..."
if ! command -v brew &>/dev/null; then
    echo "Homebrew không được cài đặt. Vui lòng cài đặt Homebrew trước."
    echo "Xem hướng dẫn tại: https://brew.sh"
    exit 1
fi

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d "venv" ]; then
    echo "🚀 Kích hoạt môi trường ảo..."
    source venv/bin/activate
else
    echo "Tạo môi trường ảo mới..."
    python3 -m venv venv
    source venv/bin/activate

    echo "Cài đặt các thư viện cần thiết..."
    pip install --upgrade pip setuptools wheel

    # Cài đặt từ requirements.txt nếu tồn tại
    if [ -f "requirements.txt" ]; then
        echo "Cài đặt các thư viện từ requirements.txt..."
        pip install -r requirements.txt
    else
        echo "Cài đặt các thư viện mặc định..."
        pip install selenium webdriver-manager pandas numpy pyspark matplotlib seaborn streamlit pyngrok ngrok folium plotly
    fi
fi

# Chuyển đến thư mục chứa script này
cd "$(dirname "$0")"

# Đảm bảo thư mục Demo tồn tại
if [ ! -d "Demo" ]; then
    echo "❌ Không tìm thấy thư mục Demo. Vui lòng kiểm tra lại cấu trúc dự án."
    exit 1
fi

echo "🌐 Bạn muốn chạy ứng dụng với ngrok để tạo URL public không? (y/n)"
read use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    # Kiểm tra xem file .env có tồn tại và có chứa NGROK_TOKEN
    if [ -f ".env" ] && grep -q "NGROK_TOKEN=" ".env"; then
        # Đọc token từ file .env
        ngrok_token=$(grep "NGROK_TOKEN=" ".env" | cut -d'=' -f2)

        # Kiểm tra xem token có giá trị hay không
        if [ -z "$ngrok_token" ]; then
            echo "🔑 Không tìm thấy token trong file .env. Vui lòng nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
            read -s ngrok_token
            # Cập nhật file .env với token mới
            sed -i '' "s/NGROK_TOKEN=/NGROK_TOKEN=$ngrok_token/" .env
        else
            echo "🔑 Đã tìm thấy ngrok token trong file .env"
        fi
    else
        echo "🔑 Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
        read -s ngrok_token
        # Lưu token vào file .env nếu file tồn tại
        if [ -f ".env" ]; then
            echo "NGROK_TOKEN=$ngrok_token" >>.env
        else
            echo "NGROK_TOKEN=$ngrok_token" >.env
        fi
    fi

    echo "⚙️ Cấu hình ngrok và khởi chạy Streamlit..."

    # Tạo file Python tạm thời để chạy ngrok
    cat >run_with_ngrok.py <<EOF

import os
import subprocess
import time
from pyngrok import ngrok

# Thiết lập ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con với file từ thư mục Demo
streamlit_process = subprocess.Popen(["streamlit", "run", "Demo/vn_real_estate_app.py"])

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
    echo "💻 Khởi chạy Streamlit trên localhost:8501..."
    streamlit run Demo/vn_real_estate_app.py
fi

# Trở về thư mục gốc và deactivate môi trường ảo khi kết thúc
cd - >/dev/null
deactivate
