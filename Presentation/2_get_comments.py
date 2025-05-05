import csv
import random
import re
import time
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException
)
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

# Hàm trả về một user agent ngẫu nhiên từ danh sách đã định trước
def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
    ]
    return random.choice(user_agents)

# Hàm thu thập đánh giá từ URL sản phẩm
def scrape_comments(product_url, quantity_sold):
    options = uc.ChromeOptions()

    # Thiết lập các tùy chọn cho Chrome
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-popup-blocking")

    # Khởi tạo undetected ChromeDriver với các tùy chọn đã chỉ định
    driver = uc.Chrome(options=options, version_main=124, use_subprocess=True)

    # Đặt kích thước cửa sổ trình duyệt và mở URL sản phẩm
    driver.set_window_size(800, 600)
    driver.get(product_url)

    try:
        title=driver.find_element(By.CSS_SELECTOR, 'h1.pdp-mod-product-badge-title').text.strip()
    except (StaleElementReferenceException, NoSuchElementException):
        print("Không tìm thấy tiêu đề")
        return

    try:
        brand = driver.find_element(By.CSS_SELECTOR, '.pdp-product-brand__brand-link').text.strip()
        print("Đang lấy dữ liệu thương hiệu sản phẩm...")
    except (StaleElementReferenceException, NoSuchElementException):
        print("Không tìm thấy thương hiệu sản phẩm.")

    try:
        price_text = driver.find_element(By.CSS_SELECTOR, '.pdp-price.pdp-price_type_normal.pdp-price_color_orange.pdp-price_size_xl').text.strip()
        # Sử dụng biểu thức chính quy để loại bỏ các ký tự không phải là số và dấu thập phân
        price_number = re.sub(r'[^\d]', '', price_text)
        price = int(price_number)
    except (StaleElementReferenceException, NoSuchElementException):
        print(f"Timeout khi cố gắng thu thập giá sản phẩm từ {product_url}")

    url = product_url
    # Cuộn trang để tải đánh giá
    driver.execute_script("window.scrollTo(0, 1500);")
    time.sleep(10)
    data = []

    try:
        for i in range(80):
            print(f"Đang thu thập đánh giá từ trang: {i+1} of {product_url}")
            comments = driver.find_elements(By.CSS_SELECTOR, 'div.item')
            if not comments:
                print("Không tìm thấy đánh giá.")
                break
            for comment in comments:
                try:
                    content = comment.find_element(By.CSS_SELECTOR, 'div.content').text.strip()
                    stars = len(comment.find_elements(By.CSS_SELECTOR, 'div.container-star.starCtn.left img.star[src="//laz-img-cdn.alicdn.com/tfs/TB19ZvEgfDH8KJjy1XcXXcpdXXa-64-64.png"]'))
                    variant = driver.find_element(By.CSS_SELECTOR, 'div.skuInfo').text.strip()
                    images = comment.find_elements(By.CSS_SELECTOR, 'div.review-image__list div.pdp-common-image')
                    has_image = 1 if images else 0
                    if stars > 0:
                        data.append([title, content, stars, variant, has_image, url, brand, price, quantity_sold])
                except (StaleElementReferenceException, NoSuchElementException):
                    continue

            try:
                # Xử lý phân trang bằng cách nhấp vào nút "Tiếp theo"
                next_btn = driver.find_element(By.CLASS_NAME, 'next-btn.next-btn-normal.next-btn-medium.next-pagination-item.next')
                if next_btn.get_attribute('disabled'):
                    print(f"Nút tiếp theo bị vô hiệu trên trang {i+1}. Thoát...")
                    break
                driver.execute_script("arguments[0].scrollIntoView();", next_btn)
                driver.execute_script("arguments[0].click();", next_btn)
                time.sleep(random.uniform(5, 10))
                driver.execute_script("window.scrollTo(0, 1800);")
            except (ElementClickInterceptedException, TimeoutException, NoSuchElementException) as e:
                print(f"Lỗi khi cố gắng nhấp vào nút tiếp theo trên trang {i+1}: {e}")
                break
    except TimeoutException:
        print(f"Timeout khi cố gắng thu thập đánh giá từ {product_url}")

    # Đóng trình duyệt và trả về dữ liệu đã thu thập
    driver.quit()
    return data

# Hàm đọc URL sản phẩm từ tệp CSV
def read_product_urls(file_path):
    df = pd.read_csv(file_path)
    return df[['url', 'quantity_sold']].to_dict(orient='records')

# Hàm ghi đánh giá vào tệp CSV
def write_comments_to_csv(comments, file_path):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(comments)
        print("Đã lưu dữ liệu đánh giá sản phẩm thành công.")

# Hàm xóa URL đã xử lý khỏi tệp CSV
def remove_processed_url(file_path, url):
    df = pd.read_csv(file_path)
    df = df[df['url'] != url]
    df.to_csv(file_path, index=False)
    print("Đã xóa dữ liệu liên kết sản phẩm đã thu thập xong đánh giá.")

# Hàm chính để đọc URL sản phẩm, thu thập đánh giá, ghi đánh giá vào tệp CSV, và xóa URL đã xử lý khỏi tệp
def main(products_info_file, comments_output_file):
    product_links = read_product_urls(products_info_file)

    for row in product_links:
        comments = scrape_comments(row['url'], row['quantity_sold'])
        if comments:
            write_comments_to_csv(comments, comments_output_file)
        remove_processed_url(products_info_file, row['url'])

# Chạy hàm main nếu script được thực thi trực tiếp
if __name__ == "__main__":
    products_info_file = 'products_info.csv'
    comments_output_file = 'comments_output.csv'  
    main(products_info_file, comments_output_file)