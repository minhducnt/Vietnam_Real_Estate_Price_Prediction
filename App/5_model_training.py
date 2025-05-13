from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p, expm1, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import os

def initialize_spark_session(app_name="RealEstateModelTraining"):
    """Khởi tạo và trả về một phiên Spark cho việc huấn luyện mô hình."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return spark

def load_processed_data(spark, data_path):
    """Đọc dữ liệu đã xử lý từ CSV hoặc HDFS."""
    df = spark.read.option("header", True).csv(data_path)
    print(f"Loaded {df.count()} records for model training")
    return df

def prepare_data_for_modeling(df):
    """Chuẩn bị dữ liệu cho việc mô hình hóa bằng cách xử lý các giá trị bị thiếu và áp dụng các phép biến đổi."""
    # Xử lý các giá trị số bị thiếu trong các cột quan trọng
    numeric_cols = ["area_m2", "floor_num", "toilet_num", "livingroom_num", "bedroom_num", "street_width_m"]

    # Tạo các cột đánh dấu dữ liệu bị thiếu và điền giá trị trung vị
    for col_name in numeric_cols:
        # Tạo cột đánh dấu dữ liệu bị thiếu
        missing_flag_col = f"{col_name}_missing_flag"
        df = df.withColumn(missing_flag_col, when(col(col_name) == -1, 1).otherwise(0))

        # Tính giá trị trung vị cho các giá trị không bị thiếu
        median_val = df.filter(col(col_name) != -1).approxQuantile(col_name, [0.5], 0.01)[0]

        # Thay thế các giá trị bị thiếu bằng giá trị trung vị
        df = df.withColumn(col_name, when(col(col_name) == -1, median_val).otherwise(col(col_name)))

    # Áp dụng phép biến đổi logarit cho giá (để giảm độ lệch)
    df = df.withColumn("price_log", log1p(col("price_per_m2")))

    print("Data prepared for modeling")
    return df

def create_model_pipeline(categorical_cols, numeric_cols, binary_cols):
    """Tạo pipeline ML cho việc xử lý đặc trưng và huấn luyện mô hình."""
    # StringIndexer cho các đặc trưng phân loại
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
        for col_name in categorical_cols
    ]

    # OneHotEncoder cho các đặc trưng phân loại đã được lập chỉ mục
    encoders = [
        OneHotEncoder(
            inputCol=f"{col_name}_index",
            outputCol=f"{col_name}_vec"
        )
        for col_name in categorical_cols
    ]

    # Kết hợp tất cả các cột đặc trưng
    feature_cols = (
        [f"{col}_vec" for col in categorical_cols] +
        numeric_cols +
        binary_cols
    )

    # VectorAssembler để kết hợp tất cả các đặc trưng thành một vector duy nhất
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # StandardScaler để chuẩn hóa đặc trưng
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )

    # Tạo các giai đoạn pipeline
    stages = indexers + encoders + [assembler, scaler]

    # Trả về pipeline
    return Pipeline(stages=stages)

def split_data(df, train_ratio=0.8, seed=42):
    """Chia dữ liệu thành tập huấn luyện và tập kiểm thử."""
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    print(f"Training set: {train_df.count()} records")
    print(f"Test set: {test_df.count()} records")
    return train_df, test_df

def train_models(train_df, test_df, feature_col="scaled_features", label_col="price_log"):
    """Huấn luyện và đánh giá các mô hình hồi quy khác nhau."""
    # Hồi quy tuyến tính
    lr = LinearRegression(featuresCol=feature_col, labelCol=label_col, maxIter=100)
    lr_model = lr.fit(train_df)
    lr_preds = lr_model.transform(test_df)

    # Cây quyết định
    dt = DecisionTreeRegressor(featuresCol=feature_col, labelCol=label_col, maxDepth=10)
    dt_model = dt.fit(train_df)
    dt_preds = dt_model.transform(test_df)

    # Rừng ngẫu nhiên
    rf = RandomForestRegressor(
        featuresCol=feature_col,
        labelCol=label_col,
        numTrees=200,
        maxDepth=8,
        seed=42
    )
    rf_model = rf.fit(train_df)
    rf_preds = rf_model.transform(test_df)

    # Gradient Boosted Trees
    gbt = GBTRegressor(
        featuresCol=feature_col,
        labelCol=label_col,
        maxIter=200,
        maxDepth=6,
        seed=42
    )
    gbt_model = gbt.fit(train_df)
    gbt_preds = gbt_model.transform(test_df)

    # Create a dictionary of models and predictions
    models = {
        "Linear Regression": (lr_model, lr_preds),
        "Decision Tree": (dt_model, dt_preds),
        "Random Forest": (rf_model, rf_preds),
        "Gradient Boosted Trees": (gbt_model, gbt_preds)
    }

    print("Models trained successfully")
    return models

def evaluate_model(pred_df, actual_col="price_per_m2", pred_col="prediction"):
    """Calculate evaluation metrics for a regression model."""
    # Convert log predictions to original scale
    pred_df = pred_df.withColumn("predicted_price", expm1(pred_col))

    # Evaluate using different metrics
    evaluator_rmse = RegressionEvaluator(labelCol=actual_col, predictionCol="predicted_price", metricName="rmse")
    evaluator_mse = RegressionEvaluator(labelCol=actual_col, predictionCol="predicted_price", metricName="mse")
    evaluator_mae = RegressionEvaluator(labelCol=actual_col, predictionCol="predicted_price", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=actual_col, predictionCol="predicted_price", metricName="r2")

    # Return metrics as a dictionary
    return {
        "RMSE": evaluator_rmse.evaluate(pred_df),
        "MSE": evaluator_mse.evaluate(pred_df),
        "MAE": evaluator_mae.evaluate(pred_df),
        "R2": evaluator_r2.evaluate(pred_df)
    }

def save_best_model(model, pipeline_model, output_dir):
    """Save the best performing model and pipeline to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the pipeline model (for feature transformation)
    pipeline_path = os.path.join(output_dir, "pipeline_model")
    pipeline_model.save(pipeline_path)

    # Save the regression model
    model_path = os.path.join(output_dir, "regression_model")
    model.save(model_path)

    print(f"Model saved to {output_dir}")

    # Create a zip file for easier distribution
    shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"Model archived as {output_dir}.zip")

    return output_dir

def plot_feature_importance(model, feature_names, output_file=None):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'featureImportances'):
        importances = model.featureImportances.toArray()

        # Create a DataFrame for the feature importances
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort by importance
        fi_df = fi_df.sort_values('Importance', ascending=False).head(20)

        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(fi_df['Feature'], fi_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file)
            print(f"Feature importance plot saved to {output_file}")
        else:
            plt.show()
    else:
        print("This model doesn't support feature importance")

def train_real_estate_model(data_path, output_dir="model"):
    """Main function to train the real estate price prediction model."""
    # Initialize Spark
    spark = initialize_spark_session()

    try:
        # Load processed data
        df = load_processed_data(spark, data_path)

        # Prepare data for modeling
        df = prepare_data_for_modeling(df)

        # Define feature columns
        categorical_cols = ["category", "direction", "liability", "district", "city_province"]
        numeric_cols = ["area_m2", "floor_num", "toilet_num", "livingroom_num", "bedroom_num", "street_width_m"]
        binary_cols = [f"{col}_missing_flag" for col in numeric_cols]

        # Create feature engineering pipeline
        pipeline = create_model_pipeline(categorical_cols, numeric_cols, binary_cols)

        # Split data
        train_df, test_df = split_data(df)

        # Fit pipeline on training data
        pipeline_model = pipeline.fit(train_df)

        # Transform both training and test data
        train_transformed = pipeline_model.transform(train_df)
        test_transformed = pipeline_model.transform(test_df)

        # Train models
        models = train_models(train_transformed, test_transformed)

        # Evaluate models
        results = {}
        for name, (model, preds) in models.items():
            results[name] = evaluate_model(preds)

        # Print evaluation results
        print("Model Evaluation Results:")
        for name, metrics in results.items():
            print(f"{name}: RMSE={metrics['RMSE']:.2f}, MSE={metrics['MSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.4f}")

        # Find the best model based on R²
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        best_model, _ = models[best_model_name]

        print(f"Best model: {best_model_name}")

        # Save the best model
        save_best_model(best_model, pipeline_model, output_dir)

        # Plot feature importance for tree-based models
        if best_model_name in ["Decision Tree", "Random Forest", "Gradient Boosted Trees"]:
            # Get feature names from the pipeline
            feature_names = []
            for stage in pipeline_model.stages:
                if isinstance(stage, VectorAssembler):
                    feature_names = stage.getInputCols()
                    break

            plot_feature_importance(
                best_model,
                feature_names,
                output_file=f"{output_dir}/feature_importance.png"
            )

        return best_model, pipeline_model, results

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    # Path to processed data
    data_path = "processed_data/part-00000-*.csv"  # Adjust based on your file naming

    # Output directory for model
    output_dir = "real_estate_model"

    # Train the model
    train_real_estate_model(data_path, output_dir)
