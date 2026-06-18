import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Customer Segmentation App", layout="centered")
st.title("🛍️ Mall Customer Segmentation Dashboard")
st.write("Input customer metrics below to instantly discover their segment and view their position on the cluster map.")

@st.cache_resource
def load_artifacts():
    with open("CustomerSegmentation.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    scaler = artifacts["scaler"]
    X_scaled = artifacts["X_scaled"]
    labels = artifacts["labels"]
except FileNotFoundError:
    st.error("❌ 'CustomerSegmentation.pkl' not found. Run your training script first!")
    st.stop()

st.subheader("👤 Customer Profile Details")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=70, value=30)
with col2:
    income = st.slider("Annual Income (k$)", min_value=15, max_value=140, value=50)
    spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Categorize Customer", type="primary"):
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })
    
    processed_data = preprocessor.transform(input_df)
    scaled_data = scaler.transform(processed_data)
    cluster_id = model.predict(scaled_data)[0]
    
    st.success(f"🏆 Target Segment Assigned: **Cluster {cluster_id + 1}**")
    
    insights = {
        1: "👵 **Senior/Mature Savers (Cluster 1)**: Stable, older demographic with conservative, moderate spending habits.",
        2: "👨‍👩‍👦 **Middle-Aged Moderate (Cluster 2)**: Standard middle-aged earners with practical, careful spending patterns.",
        3: "🌟 **Young High-Spenders (Cluster 3)**: Young, active shoppers with decent income who love to buy trends.",
        4: "💎 **VIP Elite Shoppers (Cluster 4)**: Your highest-value group! Young-adult high earners who spend heavily.",
        5: "💰 **Affluent Minimalists (Cluster 5)**: High-income individuals who rarely spend. Needs high-utility deals.",
        6: "🎒 **Young Budget-Conscious (Cluster 6)**: Lower income, but eager to spend on accessible items."
    }
    st.info(insights.get(cluster_id + 1))

    # --- LIVE 2D INTERACTIVE GRAPH (INCOME VS SPENDING) ---
    st.subheader("📊 Live Position on Customer Map")
    
    # Reconstruct columns from saved scaled data (Income is index 2, Spending is index 3)
    plot_df = pd.DataFrame({
        "Annual Income": X_scaled[:, 2],
        "Spending Score": X_scaled[:, 3],
        "Cluster": [f"Cluster {l+1}" for l in labels]
    })
    
    # Create the background scatter map
    fig = px.scatter(
        plot_df, x="Annual Income", y="Spending Score", color="Cluster",
        title="Customer Spaces (Income vs Spending Scaling)",
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.5
    )
    
    # Overlay the newly predicted customer point as a large gold star
    fig.add_scatter(
        x=[scaled_data[0, 2]], y=[scaled_data[0, 3]],
        mode="markers",
        marker=dict(color="Gold", size=18, symbol="star", line=dict(color="black", width=2)),
        name="Current Customer"
    )
    
    st.plotly_chart(fig, use_container_width=True)
