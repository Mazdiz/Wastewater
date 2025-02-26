import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# Streamlit App Title
st.title("Real-Time Wastewater Monitoring Dashboard")

# File Upload
st.sidebar.header("Upload Wastewater Data (Excel)")
uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["xlsx"], help="Limit 200MB per file")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.set_index(df.columns[0], inplace=True)
    st.sidebar.success("File Uploaded Successfully!")

    # Data Preprocessing
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Live Data Preview
    st.subheader("Live Data Preview")
    st.dataframe(df.style.format(precision=2))

    # Select Parameters for Visualization
    st.subheader("Real-Time Data Visualization")
    params = st.multiselect("Select Parameters to Visualize", numeric_columns, default=numeric_columns[:2])

    # Line Chart for Trends
    fig1 = px.line(df, x=df.index, y=params, title=f"Trend of Selected Parameters", labels={"value": "Measured Value", "variable": "Parameter"})
    st.plotly_chart(fig1)

    # Histogram for Distribution
    for param in params:
        fig2 = px.histogram(df, x=param, nbins=20, title=f"Distribution of {param}", marginal="box")
        st.plotly_chart(fig2)

    # Box Plot for Outliers
    fig3 = px.box(df, y=params, title=f"Outliers in Selected Parameters")
    st.plotly_chart(fig3)

    # Heatmap - Correlation Analysis
    st.subheader("Heatmap - Correlation Between Parameters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=1)
    st.pyplot(fig)

    # Clustering Analysis
    st.subheader("Water Quality Clustering Analysis")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_columns])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    fig4 = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], color=df['Cluster'].astype(str), title="Cluster Analysis")
    st.plotly_chart(fig4)

    # WHO/EPA Compliance Report
    st.subheader("WHO/EPA Compliance Report")
    who_limits = {'pH': (6.5, 8.5), 'Nitrate (mg/l)': (0, 10), 'Lead (mg/l)': (0, 0.01)}
    compliance_report = {}
    comparison_table = []
    for key, limits in who_limits.items():
        if key in df.columns:
            non_compliant = df[(df[key] < limits[0]) | (df[key] > limits[1])].shape[0]
            compliance_report[key] = f"{non_compliant} samples exceed WHO limits"
            comparison_table.append([key, df[key].mean(), limits[0], limits[1]])
    
    st.json(compliance_report)
    st.subheader("Measured Values vs WHO Limits")
    comparison_df = pd.DataFrame(comparison_table, columns=["Parameter", "Measured Average", "WHO Min", "WHO Max"])
    st.dataframe(comparison_df)

    # Time Series Forecasting (ARIMA)
    st.subheader("Predictive Analysis - Time Series Forecasting")
    forecast_param = st.selectbox("Select Parameter for Forecasting", numeric_columns)
    model = ARIMA(df[forecast_param], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    st.line_chart(pd.Series(forecast, index=range(len(df), len(df)+5)))

    # Alerts System
    st.subheader("Alerts & Notifications")
    alert_thresholds = {'pH': 9, 'Lead (mg/l)': 0.01, 'Nitrate (mg/l)': 10}
    alerts = []
    for key, limit in alert_thresholds.items():
        if key in df.columns and (df[key] > limit).any():
            alerts.append(f"Warning: {key} exceeded safe limits!")
    if alerts:
        st.warning("\n".join(alerts))
    else:
        st.success("No alerts. All parameters within safe limits!")

    # Export Data
    st.subheader("Export Report")
    st.download_button("Download CSV", df.to_csv(index=True).encode('utf-8'), "water_quality_report.csv", "text/csv")


