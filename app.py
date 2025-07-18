import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


# Streamlit Page Configuration
st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Using RFM Analysis")

st.markdown("""
This tool helps identify customer behavior patterns using **RFM Analysis** — Recency (How recent), Frequency (How often), and Monetary (How much they spend).
It classifies customers into meaningful segments like **Champions**, **Loyal Customers**, and **At Risk**, and recommends marketing actions for each.

# Required Format of Uploaded File
Your file must include the following columns:
- `InvoiceNo`
- `InvoiceDate`
- `CustomerID`
- `Quantity`
- `UnitPrice`
- (optional) `Country`
""")

# Upload CSV File
st.sidebar.header("📤 Upload CSV File")
data_file = st.sidebar.file_uploader("Upload your transactions CSV", type=['csv'])

# Load fallback dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Load data (uploaded or fallback)
if data_file:
    df = pd.read_csv(data_file, encoding='ISO-8859-1')
else:
    st.sidebar.warning("⚠️ No file uploaded. Using default dataset.")
    df = load_data("data.csv")

# Validate required columns
required_cols = {'InvoiceNo', 'Quantity', 'UnitPrice', 'InvoiceDate', 'CustomerID'}
if not required_cols.issubset(df.columns):
    st.error(f"❌ Uploaded data missing columns: {required_cols - set(df.columns)}")
    st.stop()

# Preprocessing
st.subheader("🔧 Data Preprocessing")
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM Calculation
rfm_ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (rfm_ref_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# RFM Scoring (1–5)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Customer Segmentation

def segment_customer(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Potential Loyalist'
    elif r >= 3 and f <= 2:
        return 'Recent Customers'
    elif r <= 2 and f >= 4:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# Marketing Recommendations
rfm['Suggested Action'] = rfm['Segment'].map({
    'Champions': 'Offer VIP loyalty reward',
    'Loyal Customers': 'Upsell premium products',
    'Potential Loyalist': 'Send exclusive preview',
    'Recent Customers': 'Welcome email & offer',
    'At Risk': 'Send win-back campaign',
    'Lost': 'Big discount or let go',
    'Others': 'General promotions'
})

# Merge optional Country
if 'Country' in df.columns:
    rfm = rfm.merge(df[['CustomerID', 'Country']].drop_duplicates(), on='CustomerID', how='left')



#  RFM Table & Segment Summary

st.subheader("📋 Full RFM Table")
st.dataframe(rfm.head(20))

st.subheader("📊 Segment Distribution")

# Create a smaller figure
fig1, ax1 = plt.subplots(figsize=(5, 2))

# Create the count plot
sns.countplot(
    data=rfm,
    x='Segment',
    order=rfm['Segment'].value_counts().index,
    palette='Set2',
    ax=ax1
)

# Rotate and resize x-axis labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=5)

# Resize y-axis tick labels
ax1.tick_params(axis='y', labelsize=8)

# Optional: resize axis titles if you add them
ax1.set_xlabel("Segment", fontsize=0.5)
ax1.set_ylabel("Count", fontsize=9)

st.pyplot(fig1)


st.subheader("📋 Segment Summary Table")
segment_summary = rfm.groupby("Segment").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).rename(columns={"CustomerID": "Count"}).sort_values(by="Count", ascending=False)
st.dataframe(segment_summary.style.background_gradient(cmap='YlGnBu').format("{:.2f}"))




# Geo Map 

if 'Country' in rfm.columns:
    st.subheader("🌍 Segment by Country")
    fig_map = px.choropleth(
        rfm,
        locations='Country',
        locationmode='country names',
        color='Segment',
        title='Customer Segment by Country',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig_map, use_container_width=True)



#Additional Insights

st.subheader("📈 Monetary vs Frequency (Positive Only)")
rfm_positive = rfm[rfm['Monetary'] > 0]
fig2 = px.scatter(
    rfm_positive,
    x='Frequency', y='Monetary', color='Segment', size='Monetary',
    hover_data=['CustomerID'],
    title='💰 Frequency vs. Monetary Value',
    color_discrete_sequence=px.colors.qualitative.Vivid
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📉 Recency Spread Across Segments")
fig3 = px.box(rfm, x='Segment', y='Recency', color='Segment', title="📦 Recency by Segment",
              color_discrete_sequence=px.colors.qualitative.Prism)
st.plotly_chart(fig3, use_container_width=True)


# Export Options (Left Panel)

st.sidebar.header("📥 Download Reports")
st.sidebar.download_button("⬇️ RFM Table", rfm.to_csv(index=False), file_name="rfm_segments.csv")
st.sidebar.download_button("⬇️ Segment Summary", segment_summary.to_csv(), file_name="rfm_summary.csv")

st.success("✅ Analysis complete! Ready for download.")
