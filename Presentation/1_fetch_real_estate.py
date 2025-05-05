import time
import random
import csv
import re

from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException, ElementClickInterceptedException
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_random_user_agent():
    """Hàm này trả về một chuỗi ngẫu nhiên là một user-agent web."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15"
    ]
    return random.choice(user_agents)

# Hàm để chuyển đổi chuỗi thành số nguyên
def convert_sold_text_to_number(sold_text):
    # Loại bỏ phần "Đã Bán" khỏi chuỗi
    sold_text = sold_text.replace(" Đã Bán", "")
    # Sử dụng regex để tìm phần số và phần đơn vị (nếu có)
    match = re.match(r'(\d+(\.\d+)?)(K?)', sold_text)
    if match:
        number = float(match.group(1))
        unit = match.group(3)
        if unit == 'K':
            number *= 1000
        return int(number)
    return 0

def get_product_info(category_url):
    """Hàm này nhận một URL của một danh mục sản phẩm và trả về một danh sách các liên kết sản phẩm từ trang web."""
    # Cài đặt tùy chọn cho trình duyệt Chrome
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={get_random_user_agent()}")  # Sử dụng user-agent ngẫu nhiên

    options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Loại bỏ thông báo lỗi không cần thiết
    options.add_argument("--start-maximized")  # Mở cửa sổ trình duyệt ở chế độ tối đa hóa

    # Khởi tạo trình duyệt Chrome
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(category_url)  # Mở URL của danh mục sản phẩm
    time.sleep(random.uniform(5, 10))  # Chờ một khoảng thời gian ngẫu nhiên từ 5 đến 10 giây

    products_info = []
    try:
        for i in range(100):  # Lặp lại quá trình tìm kiếm liên kết qua nhiều trang
            product_elements = driver.find_elements(By.CSS_SELECTOR, "div.Bm3ON[data-qa-locator='product-item']") 

            for elem in product_elements:
                # Lấy link sản phẩm
                try:
                    link_element = elem.find_element(By.CSS_SELECTOR, "a[href*='/products/']") # Tìm tất cả các phần tử liên kết sản phẩm
                    link = link_element.get_attribute("href")
                    print(link)
                except (StaleElementReferenceException, NoSuchElementException) as e:
                    print(e)
                    continue

                # Lấy số lượng đã bán
                try:
                    sold_element = elem.find_element(By.CSS_SELECTOR, "span._1cEkb span") # Tìm tất cả các phần tử số lượng đã bán của sản phẩm
                    sold_text = sold_element.text
                    quantity_sold = convert_sold_text_to_number(sold_text)
                    print(quantity_sold)
                except (StaleElementReferenceException, NoSuchElementException) as e:
                    print(e)
                    continue

                products_info.append([link, quantity_sold])

            driver.execute_script("window.scrollTo(0, 4000);") # Cuộn trang xuống
            time.sleep(random.uniform(5, 10))  # Chờ một khoảng thời gian ngẫu nhiên từ 5 đến 10 giây

            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, '.ant-pagination-next .ant-pagination-item-link')  # Tìm nút "Next"
                driver.execute_script("arguments[0].scrollIntoView();", next_btn)
                driver.execute_script("arguments[0].click();", next_btn)  # Nhấp vào nút "Next"
                time.sleep(random.uniform(5, 10))  # Chờ một khoảng thời gian ngẫu nhiên từ 5 đến 10 giây
                driver.execute_script("window.scrollTo(0, 3000);")  # Cuộn trang xuống
            except (ElementClickInterceptedException, TimeoutException, NoSuchElementException) as e:
                print(f"Lỗi khi cố gắng nhấp vào nút tiếp theo trên trang {i+1}: {e}") # Xử lý lỗi nếu có
                break

    except (TimeoutException, NoSuchElementException):
        print(f"Timeout khi cố gắng thu thập dữ liệu sản phẩm")  # Xử lý lỗi nếu có
        driver.quit()  # Đóng trình duyệt

    driver.quit()  # Đóng trình duyệt khi hoàn thành
    return products_info  # Trả về danh sách các dữ liệu sản phẩm

category_url = "https://www.lazada.vn/cham-soc-suc-khoe-va-lam-dep/"
products_info = get_product_info(category_url) # Gọi hàm để thu thập các liên kết và số lượng đã bán của sản phẩm từ URL
filename = "products_info.csv"

with open(filename, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["url", "quantity_sold"])  # Ghi dòng tiêu đề
    writer.writerows(products_info)  # Ghi dữ liệu từ danh sách

print("Dữ liệu đã được lưu vào file", filename)