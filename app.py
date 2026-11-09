from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")


@st.cache_data
def load_data():
    return pd.read_csv(Path(__file__).parent / "Dataset_3.csv")


df = load_data()

# ---- Sidebar controls ----
st.sidebar.header("Filters")
inc_min = int(df["Annual Income (k$)"].min())
inc_max = int(df["Annual Income (k$)"].max())
income_range = st.sidebar.slider("Annual Income (k$)", inc_min, inc_max, (inc_min, inc_max))
score_range = st.sidebar.slider("Spending Score (1-100)", 1, 100, (1, 100))
k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
gender = st.sidebar.multiselect(
    "Gender", df["Gender"].unique(), default=list(df["Gender"].unique())
)

# ---- Clustering on the full dataset ----
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

# ---- Apply filters ----
filtered = df[
    df["Annual Income (k$)"].between(*income_range)
    & df["Spending Score (1-100)"].between(*score_range)
    & df["Gender"].isin(gender)
]

# ---- KPIs ----
c1, c2, c3 = st.columns(3)
c1.metric("Customers shown", len(filtered))
c2.metric("Avg income (k$)", f"{filtered['Annual Income (k$)'].mean():.1f}")
c3.metric("Avg spending score", f"{filtered['Spending Score (1-100)'].mean():.1f}")

# ---- Scatter plot ----
fig = px.scatter(
    filtered,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    hover_data=["Age", "Gender"],
    title="Customer Segments",
)
st.plotly_chart(fig, use_container_width=True)

# ---- Cluster summary ----
st.subheader("Cluster summary")
summary = filtered.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
summary["Size"] = filtered["Cluster"].value_counts()
st.dataframe(summary)



# ---- Predict a new customer ----
st.sidebar.header("Predict a new customer")
new_inc = st.sidebar.number_input("Annual Income (k$)", 0, 200, 60, key="new_inc")
new_score = st.sidebar.number_input("Spending Score", 1, 100, 50, key="new_score")
if st.sidebar.button("Find segment"):
    
    pred = kmeans.predict(scaler.transform([[new_inc, new_score]]))[0]
    st.sidebar.success(f"This customer belongs to Cluster {pred}")