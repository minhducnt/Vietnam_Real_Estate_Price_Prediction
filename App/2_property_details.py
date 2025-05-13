import pandas as pd
import csv
import random
import time
import re
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException
)
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
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element.text
    except (NoSuchElementException, StaleElementReferenceException, TimeoutException):
        return ""

def get_nearby_amenities(driver):
    """Trích xuất các tiện ích lân cận như trường học, bệnh viện và chợ."""
    amenities = {
        "schools": [],
        "hospitals": [],
        "markets": [],
        "parks": [],
        "bus_stations": []
    }

    # Try to find amenities section - this is a hypothetical structure
    try:
        # Schools
        schools_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'nearby-schools')]/div[contains(@class, 'item')]")
        for school in schools_elements:
            name = school.find_element(By.XPATH, ".//div[contains(@class, 'name')]").text
            distance = school.find_element(By.XPATH, ".//div[contains(@class, 'distance')]").text
            amenities["schools"].append({"name": name, "distance": distance})

        # Similar for other amenities...
    except:
        # If not found, return empty results
        pass

    return amenities

def get_property_details(property_url):
    """Thu thập thông tin chi tiết cho một bất động sản bao gồm đặc điểm, tiện ích, v.v."""
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 1200")
    options.add_argument('--disable-dev-shm-usage')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(property_url)
        time.sleep(random.uniform(2, 5))

        # Basic information
        details = {
            "url": property_url,
            "description": safe_get_text(driver, "//div[contains(@class, 'description-content')]"),
            "posting_date": safe_get_text(driver, "//div[contains(@class, 'posting-date')]"),
            "contact_name": safe_get_text(driver, "//div[contains(@class, 'contact-name')]"),
            "contact_phone": safe_get_text(driver, "//div[contains(@class, 'contact-phone')]"),
        }

        # Extract additional features (if available)
        features = []
        feature_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'features')]/ul/li")
        for feature in feature_elements:
            features.append(feature.text)
        details["features"] = "|".join(features)

        # Extract nearby amenities
        details["nearby_amenities"] = get_nearby_amenities(driver)

        # Extract images
        image_urls = []
        try:
            image_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'carousel')]/div[contains(@class, 'item')]/img")
            for img in image_elements:
                image_url = img.get_attribute("src")
                if image_url:
                    image_urls.append(image_url)
        except:
            pass
        details["image_urls"] = "|".join(image_urls)

        return details

    except Exception as e:
        print(f"Error scraping details from {property_url}: {e}")
        return None

    finally:
        driver.quit()

def read_property_urls(file_path):
    """Đọc các URL bất động sản từ file CSV."""
    df = pd.read_csv(file_path)
    return df[['link']].to_dict(orient='records')

def append_to_csv(data, file_path="property_details.csv"):
    """Thêm thông tin chi tiết bất động sản vào file CSV."""
    # Kiểm tra xem file đã tồn tại hay chưa để xác định có cần viết tiêu đề hay không
    try:
        with open(file_path, 'r', encoding='utf-8'):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Viết tiêu đề nếu file chưa tồn tại
        if not file_exists:
            writer.writerow([
                "url", "description", "posting_date", "contact_name", "contact_phone",
                "features", "image_urls"
            ])

        # Viết dữ liệu
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
    print(f"Data appended to {file_path}")

def update_status(file_path, url, status="processed"):
    """Cập nhật trạng thái của URL bất động sản trong file CSV gốc."""
    # Đọc file CSV vào DataFrame
    df = pd.read_csv(file_path)
    # Cập nhật trạng thái cho URL tương ứng (sử dụng phương thức loc để truy cập và thay đổi giá trị)
    df.loc[df['link'] == url, 'status'] = status
    # Lưu DataFrame đã cập nhật trở lại file CSV (không bao gồm chỉ số)
    df.to_csv(file_path, index=False)
    # In thông báo cập nhật trạng thái thành công
    print(f"Updated status for {url} to {status}")

def main(input_file="raw_real_estate_data.csv", output_file="property_details.csv"):
    """Hàm chính để xử lý các URL bất động sản và lấy thông tin chi tiết."""
    # Đọc danh sách các URL bất động sản từ file đầu vào
    property_links = read_property_urls(input_file)
    # In số lượng bất động sản cần xử lý
    print(f"Found {len(property_links)} properties to process")

    # Lặp qua từng URL bất động sản với chỉ số i (bắt đầu từ 0)
    for i, row in enumerate(property_links):
        # Lấy URL từ dòng dữ liệu
        url = row['link']
        # In thông tin về tiến trình xử lý
        print(f"Processing property {i+1}/{len(property_links)}: {url}")

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
            time.sleep(random.uniform(1, 3))

        except Exception as e:
            # Xử lý ngoại lệ nếu có lỗi xảy ra
            print(f"Error processing {url}: {e}")
            # Cập nhật trạng thái thành 'error' trong file đầu vào
            update_status(input_file, url, "error")

    # In thông báo hoàn thành quá trình xử lý
    print(f"Processing complete. Results saved to {output_file}")

# Kiểm tra nếu file được chạy trực tiếp (không được import)
if __name__ == "__main__":
    # Gọi hàm main để bắt đầu quá trình xử lý
    main()
