#!/bin/bash

# Script khởi chạy ứng dụng Streamlit kết hợp với ngrok
# Tác giả: Hoa
# Ngày: 13/05/2025

echo "===== BẮT ĐẦU KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Kiểm tra và cài đặt các thư viện hệ thống cần thiết
echo "Kiểm tra các thư viện hệ thống..."
if ! command -v brew &>/dev/null; then
    echo "Homebrew không được cài đặt. Vui lòng cài đặt Homebrew trước."
    echo "Xem hướng dẫn tại: https://brew.sh"
    exit 1
fi

# Cài đặt python-setuptools nếu chưa có
if ! brew list python-setuptools &>/dev/null; then
    echo "Cài đặt python-setuptools..."
    brew install python-setuptools
fi

# Kích hoạt môi trường ảo nếu tồn tại
if [ -d "venv" ]; then
    echo "Kích hoạt môi trường ảo..."
    source venv/bin/activate
else
    echo "Tạo môi trường ảo mới..."
    python3 -m venv venv
    source venv/bin/activate

    echo "Cài đặt các thư viện cần thiết..."
    pip install --upgrade pip setuptools wheel
    pip install streamlit pyngrok pyspark matplotlib seaborn plotly folium ngrok
fi

# Kiểm tra xem đã có file demo_app.py chưa
if [ ! -f "Demo/demo_app.py" ]; then
    echo "Không tìm thấy file demo_app.py. Vui lòng kiểm tra lại!"
    exit 1
fi
pip install --upgrade pip
# Kiểm tra xem người dùng muốn sử dụng ngrok không
echo ""
echo "Bạn có muốn sử dụng ngrok để tạo URL public không? (y/n)"
read use_ngrok

if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
    echo "Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
    read -s ngrok_token

    echo "Cấu hình ngrok và khởi chạy Streamlit..."

    # Tạo file python tạm để khởi chạy ngrok
    cat >run_with_ngrok.py <<EOF
import os
import subprocess
import time
from pyngrok import ngrok

# Cấu hình ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con
streamlit_process = subprocess.Popen(["streamlit", "run", "Demo/demo_app.py"])

# Khởi tạo tunnel
http_tunnel = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print("\n" + "="*50)
print(f"URL NGROK PUBLIC: {http_tunnel.public_url}")
print("Chia sẻ URL này để cho phép người khác truy cập ứng dụng của bạn")
print("="*50 + "\n")

try:
    # Giữ script chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Dọn dẹp khi người dùng nhấn Ctrl+C
    print("\nĐang dừng ứng dụng...")
    ngrok.kill()
    streamlit_process.terminate()
EOF

    # Chạy file python tạm
    python run_with_ngrok.py

    # Xóa file tạm sau khi chạy
    rm run_with_ngrok.py
else
    echo "Khởi chạy Streamlit trên localhost:8501..."
    streamlit run Demo/demo_app.py
fi
