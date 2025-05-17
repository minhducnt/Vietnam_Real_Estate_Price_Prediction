
import os
import subprocess
import time
from pyngrok import ngrok

# Thiết lập ngrok
ngrok_token = "2x2pUaSro6bqrxKTUMp5Mc9o1wg_6YsyHb4vSXytm7DvHu6tJ"
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
