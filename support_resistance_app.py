import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(page_title="Diamond Hands Predictor", page_icon="", layout="centered")

# Title and Header
st.title("Diamond Hands Predictor")
st.image("gamestop_logo.png")
st.write("""
    ## Will GameStop go to the Moon or Crash Down?  
    Enter the current price and see if your **Diamond Hands** are worth it!
""")

# Fetch data from yfinance
ticker = yf.Ticker("GME")
period = st.selectbox("Select Time Frame", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "20y", "max"])
data = ticker.history(period=period)

# Define the features used for K-means clustering
features = ['Close', 'High', 'Low', 'Open']

# Define the scaler
scaler = StandardScaler()

# Fit the scaler to the data
scaled_features = scaler.fit_transform(data[features])

# Define the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the K-means model to the scaled data
kmeans.fit(scaled_features)

# Calculate the cluster assignment for each data point
cluster_assignments = kmeans.predict(scaled_features)

# Add the cluster assignment to the data
data['Cluster'] = cluster_assignments

# Inverse transform the cluster centers
cluster_centers = kmeans.cluster_centers_
original_cluster_centers = scaler.inverse_transform(cluster_centers)

# Plot the adjusted close prices with color coding based on clusters
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['red', 'green', 'blue']
for i in range(len(data) - 1):
    ax.plot(data.index[i:i+2], data['Close'].iloc[i:i+2], color=colors[data['Cluster'].iloc[i]])

# Plot the cluster centers and annotate them
for i, center in enumerate(original_cluster_centers[:, 0]): 
    ax.axhline(y=center, linestyle='--', color=colors[i], label=f'Cluster {i+1} Center', alpha=0.7)
    ax.text(data.index[-1], center, f'{center:.2f}', va='center', ha='left', color=colors[i])

ax.set_title('GME Close Prices with Cluster Centers')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Collecting User Input
current_price = st.number_input("Current Price ($)", min_value=0.0, step=0.01, value=data['Close'].iloc[-1])

# Determine the direction trend
def determine_trend(current_price, cluster_centers):
    max_center = max(cluster_centers[:, 0])  # Find the highest cluster center
    closest_center = min(cluster_centers[:, 0], key=lambda x: abs(x - current_price))
    
    if current_price > max_center:
        return "MOONTIME!!! ", "roaring_kitty.gif"  # Replace with your fun GIF path
    elif current_price > closest_center:
        return "Heading towards support. Manage risk.", "diamond_hands.gif"  
    elif current_price < closest_center:
        return "Heading towards resistance. Can it break through?", "peepo-rocket.gif" 
    else:
        return "Currently at a cluster center. Monitor closely.", None

if st.button("Analyze Trend"):
    trend, gif_path = determine_trend(current_price, original_cluster_centers)
    st.write(f"### Trend: {trend}")
    if gif_path:
        st.image(gif_path)