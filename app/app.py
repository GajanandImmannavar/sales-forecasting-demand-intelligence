import streamlit as st
import pandas as pd
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
REPORTS_DIR = BASE_DIR / "reports"

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 EDA Charts",
        "🤖 Model Comparison",
        "📅 Future Forecast",
        "ℹ️ About Project"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":
    
        st.title("📈 Sales Forecasting Dashboard")

        st.subheader("Machine Learning & Time Series Forecasting")

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🏆 Best Model", "Prophet")
        col2.metric("📉 MAPE", "14.33%")
        col3.metric("📅 Forecast", "12 Months")
        col4.metric("🤖 Models", "3")

        st.markdown("---")

        st.header("📌 Project Objective")

        st.write("""
This project predicts future Superstore sales using multiple forecasting models.

### Models Used

- SARIMA
- Prophet
- XGBoost

The objective is to identify the most accurate forecasting model for business decision-making.

After comparing all three models, Prophet achieved the best forecasting performance.
""")

        st.success("🏆 Best Performing Model : Prophet")
# ==========================================
# DATASET OVERVIEW
# ==========================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    try:
        df = pd.read_csv(DATA_DIR / "train.csv")

        st.success("Dataset Loaded Successfully ✅")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")

        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(info, use_container_width=True)

    except FileNotFoundError:
        st.error("❌ train.csv not found inside the data folder.")
# ==========================================
# EDA CHARTS
# ==========================================

elif page == "📈 EDA Charts":

    st.title("📈 Exploratory Data Analysis")

    charts = [
    ("Total Sales by Category", CHARTS_DIR / "01_total_sales_by_category.png"),
    ("Total Sales by Region", CHARTS_DIR / "02_total_sales_by_region.png"),
    ("Monthly Sales Trend", CHARTS_DIR / "03_monthly_sales_trend.png"),
    ("Total Sales by Month", CHARTS_DIR / "04_total_sales_by_month.png"),
    ("Average Shipping Time by Region", CHARTS_DIR / "05_average_shipping_time_by_region.png"),
    ("Total Sales by Segment", CHARTS_DIR / "06_total_sales_by_segment.png"),
    ("Top 10 Products by Sales", CHARTS_DIR / "07_top_10_products_by_sales.png"),
    ("Correlation Matrix", CHARTS_DIR / "08_correlation_matrix.png"),
    ("Monthly Time Series", CHARTS_DIR / "09_monthly_time_series.png"),
    ("ACF Plot", CHARTS_DIR / "10_acf_plot.png"),
    ("PACF Plot", CHARTS_DIR / "11_pacf_plot.png"),
    ("Train-Test Split", CHARTS_DIR / "12_train_test_split.png")
]

    for title, image_name in charts:

        st.subheader(title)

        image_path = CHARTS_DIR / image_name

        if image_path.exists():
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f"{image_name} not found.")
        # ==========================================
# MODEL COMPARISON
# ==========================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    comparison = pd.DataFrame({
        "Model": ["SARIMA", "Prophet", "XGBoost"],
        "MAE": [16295.76, 10128.56, 22902.48],
        "RMSE": [21413.03, 14561.39, 26329.60],
        "MAPE (%)": [22.49, 14.33, 33.32]
    })

    st.subheader("Model Performance")

    st.dataframe(comparison, use_container_width=True)

    st.success("🏆 Best Model: Prophet")

    st.markdown("---")

    # ==========================
    # SARIMA
    # ==========================

    st.subheader("📈 SARIMA Forecast")

    sarima_chart = CHARTS_DIR / "13_sarima_actual_vs_predicted.png"

    if sarima_chart.exists():
        st.image(sarima_chart, use_container_width=True)
    else:
        st.warning("13_sarima_actual_vs_predicted.png not found.")

    st.markdown("---")

    # ==========================
    # Prophet
    # ==========================

    st.subheader("📈 Prophet Forecast")

    prophet_chart = CHARTS_DIR / "15_prophet_actual_vs_predicted.png"

    if prophet_chart.exists():
        st.image(prophet_chart, use_container_width=True)
    else:
        st.warning("15_prophet_actual_vs_predicted.png not found.")

    st.markdown("---")

    # ==========================
    # XGBoost
    # ==========================

    st.subheader("📈 XGBoost Forecast")

    xgb_chart = CHARTS_DIR / "16_xgboost_actual_vs_predicted.png"

    if xgb_chart.exists():
        st.image(xgb_chart, use_container_width=True)
    else:
        st.warning("16_xgboost_actual_vs_predicted.png not found.")
    # ==========================================
# FUTURE FORECAST
# ==========================================

elif page == "📅 Future Forecast":

    st.title("📅 Future Sales Forecast")

    st.write("""
The Prophet model was selected as the best forecasting model based on
its lowest MAE, RMSE, and MAPE values. The following chart shows the
forecasted sales for the next 12 months.
""")

    st.markdown("---")

    # ==========================
    # Forecast Chart
    # ==========================

    forecast_chart = CHARTS_DIR / "17_future_sales_forecast.png"

    if forecast_chart.exists():
        st.image(forecast_chart, use_container_width=True)
    else:
        st.warning("17_future_sales_forecast.png not found.")

    st.markdown("---")

    # ==========================
    # Forecast Table
    # ==========================

    forecast_file = REPORTS_DIR / "forecast_12_months.csv"

    if forecast_file.exists():

        forecast = pd.read_csv(forecast_file)

        st.subheader("📋 Next 12 Months Forecast")

        st.dataframe(forecast, use_container_width=True)

    else:
        st.warning("forecast_12_months.csv not found.")

    st.markdown("---")

    st.subheader("📌 Business Recommendations")

    st.write("""
✅ Increase inventory before high-demand months.

✅ Plan marketing campaigns during peak sales periods.

✅ Allocate sufficient workforce during forecasted high-sales months.

✅ Update the forecasting model regularly with new sales data.
""")
   # ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Project")

    st.markdown("## 👨‍🎓 Student Information")

    st.write("""
**Name:** Gajanand L. Immannavar

**Project Title:** Sales Forecasting using Machine Learning and Time Series Forecasting

**Internship:** Data Science Internship
""")

    st.markdown("---")

    st.markdown("## 🎯 Project Objective")

    st.write("""
The objective of this project is to forecast future Superstore sales using
multiple forecasting techniques and identify the best-performing model
for business decision-making.
""")

    st.markdown("---")

    st.markdown("## 🤖 Models Used")

    model_df = pd.DataFrame({
        "Model": ["SARIMA", "Prophet", "XGBoost"],
        "Purpose": [
            "Statistical Time Series Forecasting",
            "Business Time Series Forecasting",
            "Machine Learning Forecasting"
        ]
    })

    st.dataframe(model_df, use_container_width=True)

    st.markdown("---")

    st.markdown("## 🛠️ Technologies Used")

    tech_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Matplotlib",
            "Scikit-Learn",
            "Statsmodels",
            "Prophet",
            "XGBoost",
            "Streamlit"
        ]
    })

    st.dataframe(tech_df, use_container_width=True)

    st.markdown("---")

    st.markdown("## 🏆 Best Model")

    st.success("""
Prophet achieved the highest forecasting accuracy.

MAPE : 14.33%

MAE : 10128.56

RMSE : 14561.39
""")

    st.markdown("---")

    st.markdown("## 📌 Project Workflow")

    st.write("""
1. Data Collection

2. Data Cleaning

3. Exploratory Data Analysis

4. Feature Engineering

5. SARIMA Forecasting

6. Prophet Forecasting

7. XGBoost Forecasting

8. Model Comparison

9. Future Sales Forecast

10. Business Recommendations
""")

    st.markdown("---")

    st.info("Developed by Gajanand L. Immannavar")