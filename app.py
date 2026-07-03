import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Forecasting & Demand Intelligence System")

# -----------------------
# Load Dataset
# -----------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y"
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()

    return df

df = load_data()
st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segmentation"

    ]

)
if page == "Sales Overview":

    st.header("Sales Overview Dashboard")
    total_sales = df["Sales"].sum()

    st.metric(

        "Total Sales",

        f"${total_sales:,.2f}"

    )
    yearly_sales = df.groupby(

        "Year"

    )["Sales"].sum().reset_index()

    st.subheader("Sales by Year")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(

        yearly_sales["Year"],

        yearly_sales["Sales"]

    )

    ax.set_xlabel("Year")

    ax.set_ylabel("Sales")

    st.pyplot(fig)
    monthly_sales = df.groupby(

        "Order Date"

    )["Sales"].sum().reset_index()

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(

        monthly_sales["Order Date"],

        monthly_sales["Sales"]

    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Sales")

    st.pyplot(fig)
    # Sales by Category
    category_sales = df.groupby("Category")["Sales"].sum().reset_index()

    st.subheader("Sales by Category")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(
        category_sales["Category"],
        category_sales["Sales"]
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Sales")

    st.pyplot(fig)


    # Sales by Region
    region_sales = df.groupby("Region")["Sales"].sum().reset_index()

    st.subheader("Sales by Region")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(
        region_sales["Region"],
        region_sales["Sales"]
    )

    ax.set_xlabel("Region")
    ax.set_ylabel("Sales")

    st.pyplot(fig)


    # Filter by Category
    st.subheader("Filter by Category")

    category = st.selectbox(
        "Choose Category",
        df["Category"].unique()
    )

    filtered = df[df["Category"] == category]

    st.dataframe(filtered)   

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    option = st.radio(
        "Forecast By",
        ["Category", "Region"]
    )

    months = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    if option == "Category":
        selected = st.selectbox(
            "Select Category",
            df["Category"].unique()
        )
        filtered = df[df["Category"] == selected]
    else:
        selected = st.selectbox(
            "Select Region",
            df["Region"].unique()
        )
        filtered = df[df["Region"] == selected]

    monthly = filtered.groupby("Order Date")["Sales"].sum().resample("M").sum()

    st.subheader("Historical Monthly Sales")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly.index,
        monthly.values,
        marker="o"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)
    st.pyplot(fig)

    model = SARIMAX(
        monthly,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12)
    )

    result = model.fit(disp=False)
    forecast = result.forecast(steps=months)

    st.subheader("Forecast")

    forecast_df = pd.DataFrame({
        "Forecast Sales": forecast
    })
    st.dataframe(forecast_df)

    future_dates = pd.date_range(
        start=monthly.index[-1],
        periods=months + 1,
        freq="M"
    )[1:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        monthly.index,
        monthly.values,
        label="Historical"
    )
    ax.plot(
        future_dates,
        forecast,
        color="red",
        marker="o",
        linewidth=3,
        label="Forecast"
    )
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    weekly_sales = df.groupby("Order Date")["Sales"].sum()
    weekly_sales = weekly_sales.resample("W").sum()
    weekly_sales = weekly_sales.reset_index()

    st.subheader("Weekly Sales")
    st.dataframe(weekly_sales.head())

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = model.fit_predict(weekly_sales[["Sales"]])
    anomalies = weekly_sales[weekly_sales["Anomaly"] == -1]

    st.metric("Total Anomalies", len(anomalies))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )
    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Detected Anomalies")
    st.dataframe(anomalies[["Order Date", "Sales"]])

    st.subheader("Business Interpretation")
    st.write("""
High sales spikes may occur because of festive seasons,
special discounts,
marketing campaigns,
or sudden customer demand.

Low sales weeks may indicate inventory shortages,
delivery delays,
or reduced customer activity.
""")

elif page == "Demand Segmentation":

    st.header("📦 Product Demand Segmentation")

    total_sales = df.groupby("Sub-Category")["Sales"].sum().reset_index()
    total_sales.columns = ["Sub-Category", "Total Sales"]

    average_order = df.groupby("Sub-Category")["Sales"].mean().reset_index()
    average_order.columns = ["Sub-Category", "Average Order Value"]

    monthly_sales = df.groupby(["Sub-Category", "Year", "Month"])["Sales"].sum().reset_index()
    sales_volatility = monthly_sales.groupby("Sub-Category")["Sales"].std().reset_index()
    sales_volatility.columns = ["Sub-Category", "Sales Volatility"]

    growth = monthly_sales.groupby("Sub-Category").agg(
        First=("Sales", "first"),
        Last=("Sales", "last")
    ).reset_index()

    growth["Growth Rate"] = ((growth["Last"] - growth["First"]) / growth["First"]) * 100
    growth = growth[["Sub-Category", "Growth Rate"]]

    product_data = total_sales.merge(average_order, on="Sub-Category")
    product_data = product_data.merge(sales_volatility, on="Sub-Category")
    product_data = product_data.merge(growth, on="Sub-Category")
    product_data.fillna(0, inplace=True)

    st.subheader("Prepared Product Dataset")
    st.dataframe(product_data)

    X = product_data[[
        "Total Sales",
        "Average Order Value",
        "Sales Volatility",
        "Growth Rate"
    ]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    product_data["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    principal = pca.fit_transform(X_scaled)
    product_data["PCA1"] = principal[:, 0]
    product_data["PCA2"] = principal[:, 1]

    cluster_map = {
        0: "High Volume Stable Demand",
        1: "Growing Demand",
        2: "Low Volume High Volatility",
        3: "Declining Demand"
    }

    product_data["Cluster Name"] = product_data["Cluster"].map(cluster_map)

    st.subheader("Demand Clusters")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=product_data,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="Set2",
        s=150,
        ax=ax
    )

    for i in range(len(product_data)):
        ax.text(
            product_data["PCA1"].iloc[i],
            product_data["PCA2"].iloc[i],
            product_data["Sub-Category"].iloc[i],
            fontsize=8
        )

    st.pyplot(fig)

    st.subheader("Cluster Table")
    st.dataframe(product_data[["Sub-Category", "Cluster", "Cluster Name"]])

    st.subheader("Products per Cluster")
    st.dataframe(product_data["Cluster Name"].value_counts())

    st.subheader("Recommended Stocking Strategy")
    st.markdown("""
### 🟢 High Volume Stable Demand
Maintain high inventory levels.

### 🔵 Growing Demand
Increase inventory gradually.

### 🟠 Low Volume High Volatility
Monitor demand carefully and keep moderate inventory.

### 🔴 Declining Demand
Reduce inventory to avoid excess stock.
""")

    st.subheader("Business Recommendation")
    st.success("""
Use demand segmentation to improve inventory planning.

Focus investment on products with growing demand while reducing stock for declining products.

This helps minimize storage costs and improve customer satisfaction.
""")