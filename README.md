# Vietnam Real Estate Price Prediction

A Python notebook-based web application for predicting real estate prices in Vietnam using a self-collected dataset, Apache Spark, PySpark, and Streamlit.

![Vietnam Flag](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Flag_of_Vietnam.svg/1200px-Flag_of_Vietnam.svg.png)

## 📋 Project Overview

This project implements a complete machine learning pipeline for real estate price prediction in Vietnam:

- **Data Collection**: Web scraping from nhadat.cafeland.vn using Selenium
- **Data Processing**: Big data processing with Apache Spark (PySpark)
- **Machine Learning**: Training gradient boosted trees regression models
- **Web Application**: Interactive UI built with Streamlit
- **Deployment**: Local and cloud deployment using ngrok

## 🛠️ Technology Stack

- **Python** - Core programming language
- **Jupyter Notebook** - Development environment
- **PySpark** - Big data processing
- **Streamlit** - Web application framework
- **Selenium** - Web scraping
- **Modern UI/UX** - TailwindCSS, responsive design
- **Ngrok** - Secure tunneling for cloud deployment

## 📂 Project Structure

```
Vietnam_Real_Estate_Price_Prediction/
├── App/                             # Core application components
│   ├── 1_fetch_real_estate.py       # Web scraping with Selenium
│   ├── 2_property_details.py        # Detailed property information extraction
│   ├── 3_preprocess_data.py         # Data cleaning and preprocessing with PySpark
│   ├── 4_HDFS_storage.py            # HDFS integration for big data storage
│   ├── 5_model_training.py          # ML model training with PySpark ML
│   ├── 6_streamlit_app.py           # Streamlit web application
│   └── 7_visualize_data.py          # Data visualization components
├── Demo/                            # Demonstration application
│   └── vn_real_estate_app.py        # Complete integrated demo application
├── Data/                            # Dataset directory
│   └── Final Data Cleaned.csv       # Preprocessed dataset
├── References/                      # Reference materials
└── Docs/                            # Documentation files
```

## 🔄 Data Pipeline

The project implements an end-to-end data pipeline:

### 1. Data Collection

- Web scraping using Selenium to collect real estate listings from nhadat.cafeland.vn
- Extraction of property details including location, price, area, features, etc.
- Storage in CSV format for further processing

```python
# Example code from App/1_fetch_real_estate.py
def fetch_real_estate_listings(base_url, num_pages=5):
    """
    Lấy danh sách bất động sản từ nhadat.cafeland.vn
    """
    # Implementation details for scraping real estate listings
```

### 2. Data Preprocessing

- Cleaning and transformation using Apache Spark (PySpark)
- Feature engineering to create meaningful attributes
- Handling missing values and outliers
- Storage in HDFS for efficient processing

```python
# Example code from App/3_preprocess_data.py
def preprocess_data(input_file, output_path):
    """Main function to preprocess real estate data."""
    # Implementation of preprocessing pipeline
```

### 3. Model Training

- Feature selection and preparation
- Training using Gradient Boosted Trees, Random Forest, and Linear Regression models
- Hyperparameter tuning for optimal performance
- Model evaluation and selection based on metrics like R², RMSE, and MAE

```python
# Example code from App/5_model_training.py
def train_real_estate_model(data_path, output_dir="model"):
    """Main function to train the real estate price prediction model."""
    # Implementation of model training pipeline
```

### 4. Web Application

- Interactive Streamlit application with modern UI
- Real-time price prediction based on user inputs
- Data visualization and exploration capabilities
- Deployment with ngrok for cloud accessibility

```python
# Example code from App/6_streamlit_app.py
def predict_price(pipeline_model, regression_model, input_data):
    """Dự đoán giá sử dụng các mô hình đã được huấn luyện."""
    # Implementation of price prediction functionality
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Java 8+ (for Apache Spark)
- Apache Spark 3.0+
- PySpark
- Selenium WebDriver
- Streamlit

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Vietnam_Real_Estate_Price_Prediction.git
   cd Vietnam_Real_Estate_Price_Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Spark environment:
   ```bash
   # Ensure JAVA_HOME and SPARK_HOME environment variables are set
   export JAVA_HOME=/path/to/java
   export SPARK_HOME=/path/to/spark
   export PYSPARK_PYTHON=python3
   ```

### Running the Application

1. Collect data (optional if you already have the dataset):
   ```bash
   python App/1_fetch_real_estate.py
   ```

2. Preprocess the data:
   ```bash
   python App/3_preprocess_data.py
   ```

3. Train the model:
   ```bash
   python App/5_model_training.py
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run App/6_streamlit_app.py
   ```

5. For cloud deployment with ngrok:
   ```bash
   streamlit run Demo/vn_real_estate_app.py
   ```

## 🌟 Features

### Price Prediction

- Estimate property prices based on location, size, and other features
- Interactive form for entering property details
- Instant price predictions with confidence intervals

### Data Exploration

- Visualize price distributions across different regions
- Analyze relationships between property features and prices
- Interactive maps showing geographical price trends

### Market Analysis

- Compare prices by property type, location, and features
- Track price trends over time
- Identify key factors influencing real estate prices

## 📊 Model Performance

The Gradient Boosted Trees model achieves:
- **R²**: 0.85+ (Coefficient of Determination)
- **RMSE**: ~15% of the average price
- **MAE**: ~12% of the average price

Key factors influencing property prices:
- Location (district and city)
- Property area
- Number of bedrooms and bathrooms
- Legal status
- Street width

## 📱 User Interface

The Streamlit application features:
- Responsive design for desktop and mobile
- Modern UI with intuitive navigation
- Interactive visualizations
- Real-time price predictions

The interface is divided into four main sections:
1. **Home** - Overview and quick statistics
2. **Price Prediction** - Input form for property details
3. **Data Exploration** - Interactive visualizations and analysis
4. **About** - Project information and technical details

## 🌐 Deployment

The application can be deployed:
- **Locally** - Using Streamlit's built-in server
- **Cloud** - Using ngrok for secure tunneling

For cloud deployment:
```python
# From Demo/vn_real_estate_app.py
def run_ngrok():
    """Kết nối ứng dụng Streamlit với ngrok để tạo URL public."""
    # Implementation of ngrok deployment
```

## 🔮 Future Improvements

- **Time Series Analysis**: Incorporate historical price data
- **Advanced ML Models**: Experiment with deep learning approaches
- **Additional Data Sources**: Integrate more real estate platforms
- **Enhanced Visualizations**: Add more interactive visualization tools
- **User Authentication**: Add user profiles for saved searches and preferences

## 👥 Team

- MSSV: 1234567
- MSSV: 1234568
- MSSV: 1234569

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Data source: nhadat.cafeland.vn
- Apache Spark community
- Streamlit development team
