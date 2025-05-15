import pandas as pd
import csv
import random
import time
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_random_user_agent():
    """Trả về một chuỗi user-agent ngẫu nhiên để tránh bị phát hiện."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
    ]
    return random.choice(user_agents)

def safe_get_text(driver, xpath):
    """Trích xuất an toàn văn bản từ một phần tử bằng cách sử dụng XPath."""
    try:
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element.text
    except (NoSuchElementException, StaleElementReferenceException, TimeoutException):
        return ""

def get_nearby_amenities(driver):
    """Trích xuất các tiện ích lân cận như trường học, bệnh viện và chợ."""
    amenities = {
        "schools": [],      # Danh sách các trường học gần đó
        "hospitals": [],    # Danh sách các bệnh viện gần đó
        "markets": [],      # Danh sách các chợ/siêu thị gần đó
        "parks": [],        # Danh sách các công viên gần đó
        "bus_stations": []  # Danh sách các trạm xe buýt gần đó
    }

    # Tìm kiếm phần mô tả tiện ích lân cận - đây là cấu trúc giả định
    try:
        # Tìm các thông tin về trường học gần đó
        schools_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'nearby-schools')]/div[contains(@class, 'item')]")
        for school in schools_elements:
            # Lấy tên trường
            name = school.find_element(By.XPATH, ".//div[contains(@class, 'name')]").text
            # Lấy khoảng cách đến trường
            distance = school.find_element(By.XPATH, ".//div[contains(@class, 'distance')]").text
            # Thêm vào danh sách trường học
            amenities["schools"].append({"name": name, "distance": distance})

        # Tương tự cho các tiện ích khác (bệnh viện, chợ, công viên, trạm xe buýt)...
    except:
        # Nếu không tìm thấy, trả về kết quả trống
        pass

    return amenities

def get_property_details(property_url):
    """Thu thập thông tin chi tiết cho một bất động sản bao gồm đặc điểm, tiện ích, v.v."""
    # Cấu hình trình duyệt Chrome để tự động hóa việc truy cập web
    options = webdriver.ChromeOptions()
    # Đặt user-agent ngẫu nhiên để tránh bị website chặn
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--verbose")                # Hiển thị thông tin chi tiết hơn trong log
    options.add_argument('--no-sandbox')             # Tăng tính bảo mật
    options.add_argument('--headless')               # Chạy Chrome ẩn (điều này giúp tăng hiệu suất)
    options.add_argument('--disable-gpu')            # Tắt GPU để tránh các vấn đề về đồ họa
    options.add_argument("--window-size=1920, 1200") # Đặt kích thước cửa sổ đủ lớn để thấy các phần tử
    options.add_argument('--disable-dev-shm-usage')   # Tắt /dev/shm để tránh các vấn đề về bộ nhớ
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Tắt logging của Chrome

    # Khởi tạo trình duyệt với các cấu hình đã định
    driver = webdriver.Chrome(options=options)

    try:
        # Truy cập vào URL bất động sản
        driver.get(property_url)
        # Tạo delay ngẫu nhiên từ 2-5 giây để mô phỏng hành vi người dùng thật
        time.sleep(random.uniform(2, 5))

        # Thu thập các thông tin cơ bản của bất động sản
        details = {
            "url": property_url,                     # Đường dẫn URL của bất động sản
            "description": safe_get_text(driver, "//div[contains(@class, 'description-content')]"),  # Mô tả chi tiết
            "posting_date": safe_get_text(driver, "//div[contains(@class, 'posting-date')]"),      # Ngày đăng tin
            "contact_name": safe_get_text(driver, "//div[contains(@class, 'contact-name')]"),      # Tên liên hệ
            "contact_phone": safe_get_text(driver, "//div[contains(@class, 'contact-phone')]"),    # Số điện thoại liên hệ
        }

        # Trích xuất các đặc điểm bổ sung (nếu có)
        features = []
        # Tìm tất cả các phần tử liệt kê đặc điểm bất động sản
        feature_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'features')]/ul/li")
        # Lặp qua từng đặc điểm và thu thập nội dung
        for feature in feature_elements:
            features.append(feature.text)
        # Kết hợp các đặc điểm thành một chuỗi, ngăn cách bằng dấu "|" để dễ phân tách sau này
        details["features"] = "|".join(features)

        # Thu thập thông tin về các tiện ích lân cận
        details["nearby_amenities"] = get_nearby_amenities(driver)

        # Thu thập các hình ảnh của bất động sản
        image_urls = []
        try:
            # Tìm tất cả các phần tử hình ảnh trong carousel ảnh (cấu trúc phổ biến của trang bất động sản)
            image_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'carousel')]/div[contains(@class, 'item')]/img")
            # Lặp qua từng phần tử hình ảnh
            for img in image_elements:
                # Lấy đường dẫn source của hình ảnh
                image_url = img.get_attribute("src")
                if image_url:  # Kiểm tra nếu URL hợp lệ
                    image_urls.append(image_url)
        except:
            # Bỏ qua nếu không thể tìm thấy hoặc xử lý hình ảnh
            pass
        # Kết hợp các URL hình ảnh thành một chuỗi, ngăn cách bằng dấu "|"
        details["image_urls"] = "|".join(image_urls)

        # Trả về từ điển chứa tất cả thông tin chi tiết về bất động sản
        return details

    except Exception as e:
        # Xử lý các ngoại lệ có thể xảy ra trong quá trình thu thập dữ liệu
        print(f"Lỗi khi thu thập thông tin chi tiết từ {property_url}: {e}")
        # Trả về None để chỉ rõ quá trình thu thập thất bại
        return None

    finally:
        # Đảm bảo luôn đóng trình duyệt sau khi hoàn thành hoặc gặp lỗi
        # Điều này giúp giải phóng tài nguyên và tránh rò rỉ bộ nhớ
        driver.quit()

def read_property_urls(file_path):
    """Đọc các URL bất động sản từ file CSV."""
    # Đọc file CSV vào DataFrame của pandas
    df = pd.read_csv(file_path)
    # Chỉ lấy cột 'link' và chuyển thành định dạng danh sách từ điển
    # Mỗi từ điển đại diện cho một dòng với khóa là tên cột
    # orient='records' nghĩa là mỗi hàng sẽ trở thành một từ điển với cặp key-value
    return df[['link']].to_dict(orient='records')

def append_to_csv(data, file_path="property_details.csv"):
    """Thêm thông tin chi tiết bất động sản vào file CSV."""
    # Kiểm tra xem file CSV đã tồn tại hay chưa để quyết định có cần thêm dòng tiêu đề hay không
    # Nếu file đã tồn tại, chúng ta sẽ không thêm tiêu đề mới
    try:
        with open(file_path, 'r', encoding='utf-8'):
            file_exists = True  # File đã tồn tại
    except FileNotFoundError:
        file_exists = False     # File chưa tồn tại, cần tạo mới và viết tiêu đề

    # Mở file ở chế độ thêm (append) với encoding UTF-8 để hỗ trợ tiếng Việt
    # newline='' giúp đảm bảo định dạng dòng mới được xử lý đúng cách trên các hệ điều hành khác nhau
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Nếu file chưa tồn tại, viết dòng tiêu đề cho các cột
        if not file_exists:
            writer.writerow([
                "url",              # Đường dẫn của bất động sản
                "description",       # Mô tả chi tiết về bất động sản
                "posting_date",      # Ngày đăng thông tin bất động sản
                "contact_name",      # Tên người liên hệ
                "contact_phone",     # Số điện thoại liên hệ
                "features",          # Các đặc điểm của bất động sản (ngăn cách bằng dấu |)
                "image_urls"         # Các URL hình ảnh của bất động sản (ngăn cách bằng dấu |)
            ])

        # Viết dữ liệu chi tiết của bất động sản vào file
        writer.writerow([
            data["url"],
            data["description"],
            data["posting_date"],
            data["contact_name"],
            data["contact_phone"],
            data["features"],
            data["image_urls"]
        ])

    # In thông báo đã thêm dữ liệu vào file CSV thành công
    print(f"Dữ liệu đã được thêm vào {file_path}")

def update_status(file_path, url, status="processed"):
    """Cập nhật trạng thái của URL bất động sản trong file CSV gốc."""
    # Đọc file CSV vào DataFrame
    df = pd.read_csv(file_path)
    # Cập nhật trạng thái cho URL tương ứng (sử dụng phương thức loc để truy cập và thay đổi giá trị)
    df.loc[df['link'] == url, 'status'] = status
    # Lưu DataFrame đã cập nhật trở lại file CSV (không bao gồm chỉ số)
    df.to_csv(file_path, index=False)
    # In thông báo cập nhật trạng thái thành công
    print(f"Cập nhật trạng thái cho {url} thành {status}")

def main(input_file="raw_real_estate_data.csv", output_file="property_details.csv"):
    """Hàm chính để xử lý các URL bất động sản và lấy thông tin chi tiết."""
    # Đọc danh sách các URL bất động sản từ file đầu vào
    property_links = read_property_urls(input_file)
    # In số lượng bất động sản cần xử lý
    print(f"Đã tìm thấy {len(property_links)} bất động sản cần xử lý")

    # Lặp qua từng URL bất động sản với chỉ số i (bắt đầu từ 0)
    for i, row in enumerate(property_links):
        # Lấy URL từ dòng dữ liệu
        url = row['link']
        # In thông tin về tiến trình xử lý
        print(f"Đang xử lý bất động sản {i+1}/{len(property_links)}: {url}")

        try:
            # Lấy thông tin chi tiết từ URL bất động sản
            details = get_property_details(url)
            if details:
                # Nếu lấy thông tin thành công, thêm vào file CSV đầu ra
                append_to_csv(details, output_file)
                # Cập nhật trạng thái thành 'processed' trong file đầu vào
                update_status(input_file, url)
            else:
                # Nếu không lấy được thông tin, đánh dấu là 'failed' trong file đầu vào
                update_status(input_file, url, "failed")

            # Thêm độ trễ ngẫu nhiên từ 1-3 giây để tránh bị phát hiện là bot
            # Điều này giúp giảm thiểu nguy cơ bị chặn IP hoặc CAPTCHA
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            # Xử lý các ngoại lệ có thể xảy ra trong quá trình xử lý từng URL
            print(f"Lỗi khi xử lý {url}: {e}")
            # Cập nhật trạng thái thành 'error' trong file đầu vào để có thể thử lại sau
            update_status(input_file, url, "error")

    # In thông báo hoàn thành quá trình xử lý và đường dẫn file kết quả
    print(f"Hoàn tất xử lý. Kết quả đã được lưu vào {output_file}")

# Điểm vào của chương trình
if __name__ == "__main__":
    # Gọi hàm main để bắt đầu quá trình thu thập và xử lý dữ liệu bất động sản
    main()
