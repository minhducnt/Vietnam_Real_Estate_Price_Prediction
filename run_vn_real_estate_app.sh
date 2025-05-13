#!/bin/bash

# Script thực thi các file trong thư mục App theo trình tự
# App/1_fetch_real_estate.py => App/2_property_details.py => ... => App/6_streamlit_app.py
# Tác giả: Cascade AI
# Ngày: 13/05/2025

echo "===== KHỞI CHẠY ỨNG DỤNG DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM ====="

# Kiểm tra thư mục App
if [ ! -d "App" ]; then
    echo "Không tìm thấy thư mục App. Vui lòng kiểm tra lại cấu trúc dự án!"
    exit 1
fi

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
    pip install selenium webdriver-manager pandas numpy pyspark matplotlib seaborn streamlit pyngrok ngrok folium plotly
fi

# Hàm hiển thị menu lựa chọn
show_menu() {
    echo ""
    echo "MENU CHÍNH:"
    echo "1. Thu thập dữ liệu bất động sản (1_fetch_real_estate.py)"
    echo "2. Lấy thông tin chi tiết bất động sản (2_property_details.py)"
    echo "3. Tiền xử lý dữ liệu (3_preprocess_data.py)"
    echo "4. Lưu trữ dữ liệu trên HDFS (4_HDFS_storage.py)"
    echo "5. Huấn luyện mô hình (5_model_training.py)"
    echo "6. Khởi chạy ứng dụng Streamlit (6_streamlit_app.py)"
    echo "7. Trực quan hóa dữ liệu (7_visualize_data.py)"
    echo "8. Chạy toàn bộ quy trình (1-7)"
    echo "9. Thoát"
    echo ""
    echo "Lựa chọn của bạn (1-9): "
}

# Hàm chạy file Python
run_python_file() {
    file_name=$1
    echo "===== ĐANG THỰC THI $file_name ====="
    python "App/$file_name"

    # Kiểm tra lỗi
    if [ $? -ne 0 ]; then
        echo "Lỗi khi thực thi $file_name. Kiểm tra lại!"
        return 1
    else
        echo "Thực thi $file_name thành công!"
        return 0
    fi
}

# Hàm chạy Streamlit với Ngrok
run_streamlit_with_ngrok() {
    echo "Bạn có muốn sử dụng ngrok để tạo URL public không? (y/n)"
    read use_ngrok

    if [[ $use_ngrok == "y" || $use_ngrok == "Y" ]]; then
        echo "Nhập ngrok authtoken của bạn (đăng ký tại ngrok.com):"
        read -s ngrok_token

        echo "Cấu hình ngrok và khởi chạy Streamlit..."

        # Tạo file python tạm để khởi chạy ngrok
        cat >run_streamlit_ngrok.py <<EOF
import os
import subprocess
import time
from pyngrok import ngrok

# Cấu hình ngrok
ngrok_token = "$ngrok_token"
ngrok.set_auth_token(ngrok_token)

# Khởi chạy Streamlit trong tiến trình con
streamlit_process = subprocess.Popen(["streamlit", "run", "App/6_streamlit_app.py"])

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
        python run_streamlit_ngrok.py

        # Xóa file tạm sau khi chạy
        rm run_streamlit_ngrok.py
    else
        echo "Khởi chạy Streamlit trên localhost:8501..."
        streamlit run App/6_streamlit_app.py
    fi
}

# Hàm chạy toàn bộ quy trình
run_full_pipeline() {
    for i in {1..5} {7..7}; do
        file_name="${i}_*.py"
        python_file=$(find App -name "$file_name" -type f)
        if [ -n "$python_file" ]; then
            base_name=$(basename "$python_file")
            run_python_file "$base_name"
            if [ $? -ne 0 ]; then
                echo "Dừng quy trình do lỗi."
                return 1
            fi
        fi
    done

    # Chạy Streamlit cuối cùng
    run_streamlit_with_ngrok
    return 0
}

# Vòng lặp chính
while true; do
    show_menu
    read choice

    case $choice in
    1)
        run_python_file "1_fetch_real_estate.py"
        ;;
    2)
        run_python_file "2_property_details.py"
        ;;
    3)
        run_python_file "3_preprocess_data.py"
        ;;
    4)
        run_python_file "4_HDFS_storage.py"
        ;;
    5)
        run_python_file "5_model_training.py"
        ;;
    6)
        run_streamlit_with_ngrok
        ;;
    7)
        run_python_file "7_visualize_data.py"
        ;;
    8)
        run_full_pipeline
        ;;
    9)
        echo "Cảm ơn bạn đã sử dụng ứng dụng. Tạm biệt!"
        exit 0
        ;;
    *)
        echo "Lựa chọn không hợp lệ. Vui lòng chọn từ 1-9."
        ;;
    esac
done
