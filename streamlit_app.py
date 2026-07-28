"""
Streamlit dashboard for the Sales Forecast AI project.
"""

from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from benchmark import BenchmarkPipeline
from prediction import Predictor, Forecaster
from config.paths import SAVED_MODELS_DIR
from utils.insights import generate_insights

# Set page config
st.set_page_config(
    page_title="Sales Forecast AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (modern dark-glass theme with professional font loads)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1.25rem;
        border-left: 6px solid #3b82f6;
        padding-left: 0.75rem;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Metrics Grid */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.08);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.075em;
        margin-top: 0.5rem;
    }
    
    /* Recommendations Cards */
    .rec-card-warning {
        background-color: #fffbeb;
        border-left: 6px solid #d97706;
        color: #92400e;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.04);
    }
    
    .rec-card-info {
        background-color: #f0f9ff;
        border-left: 6px solid #0284c7;
        color: #075985;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.04);
    }
    
    .rec-card-success {
        background-color: #f0fdf4;
        border-left: 6px solid #16a34a;
        color: #166534;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.04);
    }
    
    .rec-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .rec-card-desc {
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    /* Leaderboard Styles */
    .leaderboard-row {
        display: flex;
        align-items: center;
        background: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.01);
    }
    
    .leaderboard-rank {
        font-size: 1.5rem;
        font-weight: 800;
        width: 60px;
        text-align: center;
    }
    
    .leaderboard-name {
        flex-grow: 1;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .leaderboard-metric {
        font-size: 1.1rem;
        font-weight: 700;
        width: 120px;
        text-align: right;
        color: #2563eb;
    }
    
    .leaderboard-time {
        font-size: 0.9rem;
        color: #64748b;
        width: 150px;
        text-align: right;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📈 Sales Forecast AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An advanced corporate-grade suite to train forecasting models, compare metrics, and extract business insights.</div>', unsafe_allow_html=True)

# =====================================================
# Sidebar Configuration & Navigation
# =====================================================
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Store Dataset (CSV)",
    type=["csv"],
    help="Upload sales time series dataset containing 'date', 'store', 'item', and 'sales'."
)

# Downloadable template options
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Templates")

sample_benchmark_df = pd.DataFrame({
    "date": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
    "store": [1, 1, 1, 1, 1],
    "item": [1, 1, 1, 1, 1],
    "sales": [13, 11, 14, 13, 10]
})

sample_prediction_df = pd.DataFrame({
    "date": ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
    "store": [1, 1, 1, 1, 1],
    "item": [1, 1, 1, 1, 1]
})

st.sidebar.download_button(
    label="Download Benchmark Template (with Sales)",
    data=sample_benchmark_df.to_csv(index=False).encode('utf-8'),
    file_name="benchmark_template.csv",
    mime="text/csv",
    help="Use this template for Exploratory Data Analysis and Benchmark Models."
)

st.sidebar.download_button(
    label="Download Prediction Template (without Sales)",
    data=sample_prediction_df.to_csv(index=False).encode('utf-8'),
    file_name="predict_template.csv",
    mime="text/csv",
    help="Use this template for Predict (Single-step) and Forecast & Insights."
)

st.sidebar.info(
    "💡 **Browser download blocked?**\n"
    "You can access the template files directly in your project folder:\n"
    "📁 `sales-forecast-ai/data/`"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Task",
    [
        "📊 Exploratory Data Analysis",
        "⚙️ Benchmark Models",
        "🎯 Predict (Single-step)",
        "🔮 Forecast & Insights",
    ],
)

# Shared dataset loader logic
df = None
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        temp.write(uploaded_file.getvalue())
        dataset_path = temp.name
    df = pd.read_csv(dataset_path)

# =====================================================
# Page 1: Exploratory Data Analysis
# =====================================================
if page == "📊 Exploratory Data Analysis":
    st.markdown('<div class="section-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    
    if df is not None:
        # Overview stats
        st.subheader("Dataset Summary & Key Dimensions")
        
        # Calendar column check
        has_date = "date" in df.columns
        if has_date:
            df["date"] = pd.to_datetime(df["date"])
            min_date = df["date"].min().strftime("%Y-%m-%d")
            max_date = df["date"].max().strftime("%Y-%m-%d")
            date_range_str = f"{min_date} to {max_date}"
        else:
            date_range_str = "No Date Column Found"
            
        num_stores = df["store"].nunique() if "store" in df.columns else "N/A"
        num_items = df["item"].nunique() if "item" in df.columns else "N/A"
        
        st.markdown(
            f"""
            <div class="metrics-container">
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{num_stores}</div>
                    <div class="metric-label">Unique Stores</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{num_items}</div>
                    <div class="metric-label">Unique Items</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" style="font-size: 1.25rem; padding: 0.6rem 0;">{date_range_str}</div>
                    <div class="metric-label">Date Range</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Interactive Plotly Charts
        st.markdown('<div class="section-header">Visual Trends & Seasonality</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if has_date and "sales" in df.columns:
                st.subheader("Daily Sales Volume (Aggregated)")
                daily_sales = df.groupby("date")["sales"].sum().reset_index()
                fig_trend = px.line(
                    daily_sales, x="date", y="sales",
                    labels={"sales": "Total Sales", "date": "Date"},
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_trend.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Requires 'date' and 'sales' columns to plot aggregate trend.")
                
        with col2:
            if has_date and "sales" in df.columns:
                st.subheader("Monthly Sales Seasonality")
                df["month"] = df["date"].dt.month
                monthly_sales = df.groupby("month")["sales"].mean().reset_index()
                fig_month = px.line(
                    monthly_sales, x="month", y="sales",
                    markers=True,
                    labels={"sales": "Average Sales", "month": "Month"},
                    color_discrete_sequence=["#10b981"]
                )
                fig_month.update_layout(
                    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_month, use_container_width=True)
            else:
                st.info("Requires 'date' and 'sales' columns to plot monthly seasonality.")

        col3, col4 = st.columns(2)
        
        with col3:
            if has_date and "sales" in df.columns:
                st.subheader("Weekly Seasonality Patterns")
                df["day_name"] = df["date"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekly_sales = df.groupby("day_name")["sales"].mean().reindex(day_order).reset_index()
                fig_weekly = px.bar(
                    weekly_sales, x="day_name", y="sales",
                    labels={"sales": "Average Sales", "day_name": "Day of Week"},
                    color="sales",
                    color_continuous_scale="Viridis"
                )
                fig_weekly.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.info("Requires 'date' and 'sales' columns to plot weekly seasonality.")
                
        with col4:
            if "store" in df.columns and "sales" in df.columns:
                st.subheader("Average Sales Volume per Store")
                store_sales = df.groupby("store")["sales"].mean().reset_index()
                fig_store = px.bar(
                    store_sales, x="store", y="sales",
                    labels={"sales": "Average Sales", "store": "Store ID"},
                    color="sales",
                    color_continuous_scale="Plasma"
                )
                fig_store.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_store, use_container_width=True)
            else:
                st.info("Requires 'store' and 'sales' columns to plot store-wise sales.")
                
    else:
        st.info("💡 Please upload a CSV dataset from the sidebar to visualize trends and seasonality.")

# =====================================================
# Page 2: Benchmark Models
# =====================================================
elif page == "⚙️ Benchmark Models":
    st.markdown('<div class="section-header">Interactive Model Benchmarking Suite</div>', unsafe_allow_html=True)
    
    # Advanced Sidebar Hyperparameters control
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Model Configuration")
    
    enabled_models = []
    model_kwargs = {}
    
    if st.sidebar.checkbox("Naive Baseline", value=True):
        enabled_models.append("baseline")
        
    if st.sidebar.checkbox("Linear Regression", value=True):
        enabled_models.append("linear_regression")
        
    if st.sidebar.checkbox("Random Forest", value=True):
        enabled_models.append("random_forest")
        with st.sidebar.expander("Random Forest Params"):
            rf_est = st.slider("RF Estimators", 10, 300, 100, 10)
            rf_depth = st.slider("RF Max Depth", 5, 40, 15, 5)
            model_kwargs["random_forest"] = {
                "n_estimators": rf_est,
                "max_depth": rf_depth
            }
            
    if st.sidebar.checkbox("XGBoost", value=True):
        enabled_models.append("xgboost")
        with st.sidebar.expander("XGBoost Params"):
            xgb_est = st.slider("XGB Estimators", 10, 300, 100, 10)
            xgb_depth = st.slider("XGB Max Depth", 3, 15, 6, 1)
            xgb_lr = st.slider("XGB Learning Rate", 0.01, 0.3, 0.05, 0.01)
            model_kwargs["xgboost"] = {
                "n_estimators": xgb_est,
                "max_depth": xgb_depth,
                "learning_rate": xgb_lr
            }
            
    if st.sidebar.checkbox("PyTorch NN (LSTM)", value=True):
        enabled_models.append("lstm")
        with st.sidebar.expander("Neural Network Params"):
            nn_epochs = st.slider("Epochs", 1, 20, 5, 1)
            nn_batch = st.select_slider("Batch Size", [16, 32, 64, 128], 64)
            nn_lr = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], 0.001)
            model_kwargs["lstm"] = {
                "epochs": nn_epochs,
                "batch_size": nn_batch,
                "lr": nn_lr
            }

    if df is not None:
        st.subheader("Data Overview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Sampling controls for safety
        sampled_df = df
        if len(df) > 15000:
            st.warning(f"⚠️ Dataset contains {len(df):,} rows. Training full models might take a while.")
            use_sampling = st.checkbox("Sample dataset to latest rows for high-speed benchmarking", value=True)
            sample_size = st.number_input("Maximum sample size", min_value=1000, max_value=200000, value=50000, step=10000)
            if use_sampling:
                if "date" in df.columns:
                    sampled_df = df.sort_values("date").tail(sample_size).reset_index(drop=True)
                else:
                    sampled_df = df.tail(sample_size).reset_index(drop=True)
                st.info(f"Dataset sampled to latest {sample_size:,} rows.")

        if st.button("🚀 Run Benchmark Pipeline"):
            if len(enabled_models) == 0:
                st.error("Please enable at least one forecasting model from the sidebar parameters.")
            elif "sales" not in sampled_df.columns:
                st.error("⚠️ The uploaded dataset does not contain the target column 'sales'. Benchmarking requires the target column to train and evaluate models. Please upload a training dataset (like `train.csv`).")
            else:
                with st.spinner("Executing data engineering and training selected models..."):
                    # Save sampled dataset to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_sampled:
                        sampled_df.to_csv(temp_sampled.name, index=False)
                        sampled_path = temp_sampled.name

                    pipeline = BenchmarkPipeline(
                        data_path=sampled_path,
                        enabled_models=enabled_models,
                        model_kwargs=model_kwargs
                    )
                    results = pipeline.run()

                st.success("🎉 Benchmark completed successfully!")
                
                # Visual Leaderboard
                st.markdown('<div class="section-header">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
                
                for idx, row in results.iterrows():
                    rank = idx + 1
                    badge = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    
                    st.markdown(
                        f"""
                        <div class="leaderboard-row">
                            <div class="leaderboard-rank">{badge}</div>
                            <div class="leaderboard-name">{row['Model']}</div>
                            <div class="leaderboard-metric">RMSE: {row['RMSE']:.3f}</div>
                            <div class="leaderboard-metric" style="color: #64748b; font-weight: normal;">MAE: {row['MAE']:.3f}</div>
                            <div class="leaderboard-time">Train Time: {row['Training Time (s)']:.2f}s</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Comparison plot
                st.subheader("Model Performance Comparison (RMSE)")
                fig_comp = px.bar(
                    results, x="Model", y="RMSE",
                    labels={"RMSE": "RMSE (Lower is Better)"},
                    color="RMSE",
                    color_continuous_scale="Reds_r"
                )
                fig_comp.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Feature Importance
                st.markdown('<div class="section-header">Explainable AI (Feature Importance)</div>', unsafe_allow_html=True)
                feat_imp_path = Path("reports/results/feature_importance.csv")
                if feat_imp_path.exists():
                    feat_imp = pd.read_csv(feat_imp_path)
                    st.subheader("Best Model Feature Importances")
                    
                    fig_imp = px.bar(
                        feat_imp.head(15), x="Importance", y="Feature",
                        orientation="h",
                        labels={"Importance": "Importance Score", "Feature": "Feature Name"},
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    fig_imp.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance metrics not generated. A tree-based model must be ranked best to display features explanation.")
    else:
        st.info("Please upload a CSV dataset from the sidebar to train and compare forecasting models.")

# =====================================================
# Page 3: Predict
# =====================================================
elif page == "🎯 Predict (Single-step)":
    st.markdown('<div class="section-header">Single-step Sales Prediction</div>', unsafe_allow_html=True)
    
    models = sorted([model.name for model in SAVED_MODELS_DIR.glob("*.pkl")])
    
    if len(models) == 0:
        st.warning("⚠️ No trained models found. Please run the Benchmarking pipeline first to train and save models.")
    else:
        model_name = st.selectbox("Select Model", models)
        uploaded_features = st.file_uploader("Upload Features CSV", type=["csv"], key="predict_features")
        
        if uploaded_features:
            features_df = pd.read_csv(uploaded_features)
            st.subheader("Features Preview")
            st.dataframe(features_df.head(), use_container_width=True)
            
            if st.button("Generate Predictions"):
                predictor = Predictor(SAVED_MODELS_DIR / model_name)
                predictions = predictor.predict(features_df)
                
                st.subheader("Predictions Output")
                result_df = pd.concat([features_df, predictions], axis=1)
                st.dataframe(result_df, use_container_width=True)
                
                st.download_button(
                    "📥 Download Predictions (CSV)",
                    result_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )
                
                # Plotly prediction plot
                if "Prediction" in predictions.columns:
                    st.subheader("Prediction Distribution Chart")
                    fig_pred = px.line(
                        result_df.tail(200), y="Prediction",
                        color_discrete_sequence=["#ea580c"],
                        title="Prediction Trend over Latest Features"
                    )
                    fig_pred.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

# =====================================================
# Page 4: Forecast & Insights
# =====================================================
else:
    st.markdown('<div class="section-header">Multi-step Future Forecasting & Business Insights</div>', unsafe_allow_html=True)
    
    models = sorted([model.name for model in SAVED_MODELS_DIR.glob("*.pkl")])
    
    if len(models) == 0:
        st.warning("⚠️ No trained models found. Please run the Benchmarking pipeline first to train and save models.")
    else:
        model_name = st.selectbox("Select Model for Forecasting", models)
        uploaded_future = st.file_uploader("Upload Future Features CSV", type=["csv"], key="forecast_features")
        
        if uploaded_future:
            future_df = pd.read_csv(uploaded_future)
            st.subheader("Future Features Preview")
            st.dataframe(future_df.head(), use_container_width=True)
            
            if st.button("🔮 Generate Forecast & Insights"):
                predictor = Predictor(SAVED_MODELS_DIR / model_name)
                forecaster = Forecaster(predictor)
                forecast = forecaster.forecast(future_df)
                
                st.subheader("Forecast Results")
                st.dataframe(forecast, use_container_width=True)
                
                # Interactive Plotly Forecast Chart
                st.subheader("Visualized Forecast")
                if "date" in forecast.columns:
                    fig_fc = px.line(
                        forecast, x="date", y="Forecast",
                        color_discrete_sequence=["#8b5cf6"],
                        labels={"Forecast": "Forecasted Sales", "date": "Date"}
                    )
                else:
                    fig_fc = px.line(
                        forecast, y="Forecast",
                        color_discrete_sequence=["#8b5cf6"],
                        labels={"Forecast": "Forecasted Sales"}
                    )
                fig_fc.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                
                # AI Business Insights Section
                st.markdown('<div class="section-header">💡 Business Insights & Actionable Recommendations</div>', unsafe_allow_html=True)
                
                insights = generate_insights(forecast)
                
                if len(insights) > 0:
                    summary = insights["summary"]
                    rec = insights["recommendations"]
                    
                    # Display summary metrics in visual cards
                    st.markdown(
                        f"""
                        <div class="metrics-container">
                            <div class="metric-card">
                                <div class="metric-value">{summary['total_sales']:,.0f}</div>
                                <div class="metric-label">Total Projected Sales</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{summary['avg_sales']:,.1f}</div>
                                <div class="metric-label">Daily Average Sales</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{summary['max_sales']:,.1f}</div>
                                <div class="metric-label">Peak Demand Single-day</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{summary['trend_icon']} {summary['trend_direction']}</div>
                                <div class="metric-label">Overall Demand Trend</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Demand Trend Summary:** {summary['trend_summary']}")
                    st.markdown("---")
                    
                    # Actionable Recommendations
                    st.subheader("Strategic Guidelines & Inventory Planning")
                    for r in rec:
                        card_class = "rec-card-warning" if r["type"] == "warning" else "rec-card-info" if r["type"] == "info" else "rec-card-success"
                        st.markdown(
                            f"""
                            <div class="{card_class}">
                                <div class="rec-card-title">{r['title']}</div>
                                <div class="rec-card-desc">{r['description']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                st.download_button(
                    "📥 Download Forecast Report (CSV)",
                    forecast.to_csv(index=False),
                    "forecast.csv",
                    "text/csv"
                )