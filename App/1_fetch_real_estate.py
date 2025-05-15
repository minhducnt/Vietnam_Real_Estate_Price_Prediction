import time
import random
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
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
        return driver.find_element(By.XPATH, xpath).text
    except (NoSuchElementException, StaleElementReferenceException):
        return ""

def fetch_real_estate_listings(base_url, num_pages=5):
    """
    Lấy danh sách bất động sản từ nhadat.cafeland.vn

    Tham số:
    -----------
    base_url : str
        URL cơ bản cho danh sách bất động sản
    num_pages : int
        Số trang cần thu thập

    Trả về:
    --------
    list
        Danh sách các từ điển chứa dữ liệu bất động sản
    """
    # Cấu hình tùy chọn Chrome cho trình duyệt headless
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 1200")
    options.add_argument('--disable-dev-shm-usage') # Hạn chế lỗi bộ nhớ
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Tắt thông báo log

    # Khởi tạo trình duyệt Chrome
    driver = webdriver.Chrome(options=options)

    # Thu thập các URL của danh sách bất động sản
    article_links = []

    # Lặp qua từng trang danh sách
    for page in range(1, num_pages + 1):
        url = f"{base_url}/?page={page}"
        print(f"Thu thập URL từ trang {page}: {url}")

        try:
            driver.get(url)
            time.sleep(random.uniform(3, 6))  # Độ trễ ngẫu nhiên để tránh bị phát hiện

            # Tìm tất cả các phần tử danh sách
            posts = driver.find_elements(By.XPATH, "//div[contains(@class, 'property-list')]/div[contains(@class, 'row-item')]")

            # Trích xuất liên kết từ mỗi danh sách
            for post in posts:
                try:
                    link_elem = post.find_elements(By.XPATH, ".//a")
                    link = link_elem[0].get_attribute("href") if link_elem else ""
                    article_links.append({"link": link})
                except Exception as e:
                    print(f"Lỗi xử lý danh sách: {e}")

        except Exception as e:
            print(f"Lỗi truy cập trang {page}: {e}")

    print(f"Thu thập được {len(article_links)} URL danh sách")

    # Đóng trình duyệt sau khi thu thập URL
    driver.quit()

    # Khởi tạo một trình duyệt mới để thu thập chi tiết bất động sản
    driver = webdriver.Chrome(options=options)

    # Thu thập dữ liệu chi tiết về bất động sản
    property_data = []

    # Xử lý từng URL bất động sản
    for idx, article in enumerate(article_links):
        url = article['link']
        print(f"[{idx+1}/{len(article_links)}] Thu thập chi tiết từ: {url}")

        try:
            driver.get(url)
            time.sleep(random.uniform(2, 4))  # Độ trễ ngẫu nhiên

            # Trích xuất chi tiết bất động sản
            property_info = {
                "link": url,
                "title": safe_get_text(driver, "//h1[contains(@class, 'head-title')]"),
                "location": safe_get_text(driver, "//div[contains(@style, 'float:left; width:87%; padding-left: 5px;')]"),
                "price": safe_get_text(driver, "(//div[contains(@class, 'col-item')]/div[contains(@class, 'infor-note')])[2]"),
                "area": safe_get_text(driver, "(//div[contains(@class, 'col-item')]/div[contains(@class, 'infor-data')])[2]"),
                "category": safe_get_text(driver, "//div[contains(@class, 'opt-mattien')]/span[contains(@class, 'value-item')]"),
                "direction": safe_get_text(driver, "//div[contains(@class, 'opt-huongnha')]/span[contains(@class, 'value-item')]"),
                "floor_num": safe_get_text(driver, "//div[contains(@class, 'opt-sotang')]/span[contains(@class, 'value-item')]"),
                "toilet_num": safe_get_text(driver, "//div[contains(@class, 'opt-sotoilet')]/span[contains(@class, 'value-item')]"),
                "street": safe_get_text(driver, "//div[contains(@class, 'opt-duong')]/span[contains(@class, 'value-item')]"),
                "livingroom_num": safe_get_text(driver, "//div[contains(@class, 'opt-bancong')]/span[contains(@class, 'value-item')]"),
                "bedroom_num": safe_get_text(driver, "//div[contains(@class, 'opt-sopngu')]/span[contains(@class, 'value-item')]"),
                "liability": safe_get_text(driver, "//div[contains(@class, 'opt-phaply')]/span[contains(@class, 'value-item')]"),
            }

            property_data.append(property_info)

        except Exception as e:
            print(f"Lỗi thu thập chi tiết từ {url}: {e}")

    # Đóng trình duyệt
    driver.quit()

    return property_data

def save_to_csv(data, filename="raw_bat_dong_san.csv"):
    """Lưu dữ liệu đã thu thập vào file CSV trong thư mục Data."""
    import os

    # Tạo thư mục Data nếu nó chưa tồn tại
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
    os.makedirs(data_dir, exist_ok=True)

    # Đường dẫn đầy đủ đến file
    file_path = os.path.join(data_dir, filename)

    # Lưu dữ liệu vào file CSV
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print(f"Dữ liệu đã được lưu vào {file_path}")
    return file_path

if __name__ == "__main__":
    BASE_URL = "https://nhadat.cafeland.vn/nha-dat-ban"

    # Thu thập danh sách bất động sản
    print("Bắt đầu thu thập dữ liệu bất động sản...")
    property_data = fetch_real_estate_listings(BASE_URL, num_pages=10)

    # Lưu dữ liệu vào CSV
    output_file = save_to_csv(property_data)
    print(f"Đã thu thập được {len(property_data)} danh sách bất động sản!")
    print(f"Dữ liệu đã được lưu vào {output_file}")
