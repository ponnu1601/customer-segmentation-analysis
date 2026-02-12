#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Analysis - Online Retail

# ## 1. Project Overview
# 
#  This project performs customer segmentation analysis on online retail data to identify distinct customer groups based on 
# their purchasing behavior using RFM (Recency, Frequency, Monetary) analysis.
# 
# **Objectives:**
#     - Segment customers based on purchasing behavior
#     - Identify high-value customer groups
#     - Provide actionable business recommendations
# 
# **Dataset:** Online Retail dataset (Dec 2010 - Dec 2011)
# 
# **Source:** [Kaggle - Online Retail Customer Clustering](https://www.kaggle.com/datasets/hellbuoy/online-retail-customer-clustering)
# 
# **Description:** Transnational dataset containing all transactions for a UK-based online retail company specializing in unique all-occasion gifts, with many wholesale customers.
# 
# **Business Goal:** Build RFM clustering to identify the best customer segments for targeted marketing strategies.

# ## 2. Setup

# ### 2.1 Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ### 2.2 Load Dataset

# In[2]:


# Load the dataset
df = pd.read_csv('OnlineRetail.csv', encoding = 'latin-1')


# In[3]:


# Basic dataset information
print("=== DATASET OVERVIEW === \n")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique customers: {df['CustomerID'].nunique()}")


print("\n=== SAMPLE DATA ===")
# Display first few rows
df.head()


# ## 3. Exploratory Data Analysis

# ### 3.1 Dataset Structure

# In[4]:


# Detailed dataset information
print("=== DATASET STRUCTURE ===\n")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n=== DATA TYPES & MISSING VALUES ===\n")
# Data types and missing values
df.info()


# ### 3.2 Missing Values Analysis

# In[5]:


# Check missing values
print("=== MISSING VALUES ANALYSIS ===")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})
missing_summary


# ### 3.3 Date Range Analysis

# In[6]:


# Convert and analyze dates
print("=== DATE RANGE ANALYSIS ===\n")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

# Convert InvoiceDate to datetime
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Analysis period: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")
print(f"New data type: {df['InvoiceDate'].dtype}")


# ### 3.4 Numerical Data Analysis

# In[7]:


# Analyze numerical columns
print('=== NUMERICAL DATA ANALYSIS ===')

# Basic statistics for numerical columns
numerical_cols = ['Quantity', 'UnitPrice']
df[numerical_cols].describe()


# In[8]:


# Check for negative values and zeros
print(f"Negative quantities : {(df['Quantity'] < 0).sum():,}")
print(f"Zero quantities : {(df['Quantity'] == 0).sum():,}")
print(f"Negative prices : {(df['UnitPrice'] < 0).sum():,}")
print(f"Zero prices : {(df['UnitPrice'] == 0).sum():,}")


# In[9]:


# Example for a negative quantity 
if (df['Quantity'] < 0).sum() > 0:
    print(f"\nSample negative quantities (returns):")
(df[df['Quantity'] < 0][['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice']].head(3))


# ## 4. Data Cleaning

# ### 4.1 Remove Missing CustomerIDs

# In[10]:


# Remove transactions without CustomerID 
print('=== REMOVING MISSING CUSTOMERIDs ===\n')
print(f"Before cleaning : {len(df):,} rows")

df_clean = df.dropna(subset = ['CustomerID']).copy()

print(f"After cleaning : {len(df_clean):,} rows")
print(f"Rows removed : {len(df) - len(df_clean) :,}")
print(f"Unique customers remaining : {(df_clean['CustomerID']).nunique():,}")


# ### 4.2 Remove Zero Prices

# In[11]:


# Remove transactions with zero prices 
print('=== REMOVING ZERO PRICES ===\n')
print(f"Before removing zero prices : {len(df_clean):,} rows")
print(f"Zero price transactions : {(df_clean['UnitPrice'] == 0).sum():,}")

df_clean = df_clean[df_clean['UnitPrice'] > 0]

print(f"After removing zero prices : {len(df_clean):,} rows")
print(f"Unique customers : {df_clean['CustomerID'].nunique():,}")


# ### 4.3 Check for Duplicate Transactions

# In[12]:


# Check for duplicate transactions
print('=== CHECKING FOR DUPLICATE TRANSACTIONS ===')

# Check for exact duplicates
exact_duplicates = df_clean.duplicated().sum()
print(f"Exact Duplicates : {exact_duplicates:,}")

# Check for potential duplicate transactions (same customer, product, date, quantity, price)
potential_duplicates = df_clean.duplicated(subset = ['CustomerID', 'StockCode', 'InvoiceDate', 'Quantity', 'UnitPrice']).sum()
print(f"Potential Duplicates : {potential_duplicates:,}")


# In[13]:


# Removing exact duplicates
df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates : {len(df_clean):,} rows")
print(f"Unique customers : {df_clean['CustomerID'].nunique():,}")


# ## 5. Feature Engineering

# ### 5.1 Create Transaction Amount Column

# In[14]:


# Calculate total amount for each transaction
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

#Print first few rows
df_clean.head()


# In[15]:


# Count of Positive and Negative Amounts
print(f"Posititve amount (purchases) : {(df_clean['TotalAmount'] > 0).sum():,}")
print(f"Negative amount (purchases) : {(df_clean['TotalAmount'] < 0).sum():,}")


# ### 5.2 Remove Unprofitable Customers

# In[16]:


# Calculate net spending per customer
customer_net_spending = df_clean.groupby('CustomerID')['TotalAmount'].sum()
customer_net_spending


# In[17]:


# Find customers with non-positive spending
unprofitable_customers = customer_net_spending[customer_net_spending <= 0].index
print(f"Unprofitable customers : {len(unprofitable_customers)}")


# In[18]:


# Remove unprofitable customers
df_clean = df_clean[~df_clean['CustomerID'].isin(unprofitable_customers)]
print(f"Remaining Transactions : {len(df_clean):,}")
print(f"Unique customers : {df_clean['CustomerID'].nunique():,}")


# ## 6. RFM Analysis

# ### 6.1 Calculate RFM Metrics

# In[19]:


# Set analysis date (day after the last transaction)
analysis_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days = 1)
print('Analysis date :', analysis_date)


# In[20]:


# Calculate RFM for each customer
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate' : lambda x: (analysis_date - x.max()).days,
    'InvoiceNo' : 'nunique',
    'TotalAmount' : 'sum'
}).round(2)


# In[21]:


# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()


# In[22]:


# Check for customers with zero spending
zero_customers = rfm[rfm['Monetary'] == 0.0]
print(f"Customers with exactly £0.00: {len(zero_customers)}")

if len(zero_customers) > 0:
    print(f"Removing {len(zero_customers)} customers with zero spending...")
    rfm = rfm[rfm['Monetary'] > 0].copy()
    print(f"Remaining customers: {len(rfm):,}")


# In[23]:


# RFM statistics
rfm.describe()


# ### 6.2 Top Customer Analysis

# In[24]:


# Most recent customers
print('Most Recent Customers')
(rfm.nsmallest(5, 'Recency')[['Recency', 'Frequency', 'Monetary']])


# In[25]:


# Most Frequent Customers
print('Most Frequent Customers')
(rfm.nlargest(5, 'Frequency')[['Recency', 'Frequency', 'Monetary']])


# ### 6.3 Additional Customer Metrics

# In[26]:


# Calculate additional customer metrics
customer_metrics = df_clean.groupby('CustomerID').agg({
    'TotalAmount' : ['sum', 'mean'],
    'Quantity' : 'sum',
    'InvoiceDate' : ['min', 'max'],
    'InvoiceNo' : 'nunique'
}).round(2)


# In[27]:


# Rename columns
customer_metrics.columns = ['TotalSpent', 'Ave_Transaction_value', 'TotalQuantity', 'First_Purchase', 'Last_Purchase', 'Uniqueorders']
customer_metrics.head()


# In[28]:


# Calculate date range (days between first and last purchase)
customer_metrics['PurchaseDateRange'] = (
    customer_metrics['Last_Purchase'] - customer_metrics['First_Purchase']
).dt.days

# Calculate Average Order Value (per order)
order_totals = df_clean.groupby(['CustomerID', 'InvoiceNo'])['TotalAmount'].sum()
customer_aov = order_totals.groupby('CustomerID').mean().round(2)
customer_metrics['Avg_Order_Values'] = customer_aov

customer_metrics.head()


# In[29]:


print('1. Average Order Value : ')
print(f"\t Overall AOV : £ {customer_aov.mean():.2f}")
print(f"\t AOV range : £ {customer_aov.min():.2f} - £ {customer_aov.max():.2f}")

print(f"\n2. Total Quantity Purchased :")
print(f"\t Average per customer: {customer_metrics['TotalQuantity'].mean():.0f} items")
print(f"\t Range: {customer_metrics['TotalQuantity'].min():.0f} - {customer_metrics['TotalQuantity'].max():.0f} items")

print(f"\n3. Purchase Date Range :")
print(f"\t Average customer lifespan: {customer_metrics['PurchaseDateRange'].mean():.0f} days")
print(f"\t Range: {customer_metrics['PurchaseDateRange'].min():.0f} - {customer_metrics['PurchaseDateRange'].max():.0f} days")


# ## 7. Customer Segmentation

# ### 7.1 Determine Optimal Number of Clusters

# In[30]:


# Prepare data for clustering (normalize the RFM values)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)


# In[31]:


# Test different numbers of clusters
inertias = []
silhouette_scores = []
k_range = range(2,11)

for k in k_range:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_scaled, cluster_labels))


# In[32]:


# Plot both methods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))

#Elbow method
ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number od Clusters (k)')
ax1.set_ylabel('Inertial')
ax1.set_title('Elbow Method')
ax1.grid(True)

# Silhouette score
ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score Analysis')
ax2.grid(True)

plt.tight_layout()
plt.show()


# In[33]:


for k, inertia, silhouette in zip(k_range, inertias, silhouette_scores):
    print(f"{k}\t{inertia:.2f}\t\t{silhouette:.4f}")

print(f"\nOptimal number of clusters based on:")
print(f"• Highest Silhouette Score: k={k_range[silhouette_scores.index(max(silhouette_scores))]} (score: {max(silhouette_scores):.4f})")


# ### 7.2 Apply K-Means Clustering

# In[34]:


# Based on the analysis:
# Elbow method: Shows elbow around k=4
# Silhouette score: Highest at k=4 (0.62)
optimal_clusters = 4

# Apply K-means with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm.head()


# In[35]:


# Show cluster sizes
cluster_sizes = rfm['Cluster'].value_counts().sort_index()
for i in range(optimal_clusters):
    print(f"  Cluster {i}: {cluster_sizes[i]:,} customers ({cluster_sizes[i]/len(rfm)*100:.1f}%)")


# In[36]:


# Show cluster characteristics
cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean', 
    'Monetary': 'mean'
}).round(2)
cluster_summary


# ## 8. Segment Analysis & Visualization

# ### 8.1 Customer Segment Profiles

# In[37]:


# Analyze and name customer segments based on RFM characteristics
segment_names = {
    0: "High Value Customers",
    1: "At-Rist/Lost Customers",
    2: "VIP Customers", 
    3: "Regular Customers"
}


# In[38]:


# Cluster Profiling
for cluster in range(4):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    size = len(cluster_data)
    percentage = (size / len(rfm) * 100)
    
    print(f"\nCLUSTER {cluster}: {segment_names[cluster]}")
    print(f"   Size: {size:,} customers ({percentage:.1f}%)")
    print(f"   Recency: {cluster_data['Recency'].mean():.0f} days (last purchase)")
    print(f"   Frequency: {cluster_data['Frequency'].mean():.1f} transactions")
    print(f"   Monetary: £{cluster_data['Monetary'].mean():.2f} average spending")


# ### 8.2 Segment Visualization

# #### 8.2.1 Recency vs Montary Analysis

# In[39]:


# 1. Recency vs Monetary visualization
plt.figure(figsize=(12, 8))

colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
segment_names_short = ['High-Value', 'At-Risk/Lost', 'VIP', 'Regular']

for i in range(4):
    cluster_data = rfm[rfm['Cluster'] == i]
    plt.scatter(cluster_data['Recency'], cluster_data['Monetary'], 
               c=colors[i], label=f'Cluster {i}: {segment_names_short[i]}', 
               alpha=0.7, s=50)

plt.xlabel('Recency (Days Since Last Purchase)', fontsize=12)
plt.ylabel('Monetary (Total Spending £)', fontsize=12)
plt.title('Customer Segments: Recency vs Monetary Value', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# In[40]:


cluster_3_high_recency = rfm[(rfm['Cluster'] == 0) & (rfm['Recency'] > 250)]
cluster_3_high_recency


# **Interpretation of the graph:**
# 
# This customer segmentation graph shows four distinct groups in order of priority: 
# 
# Cluster 2 (Red) - VIP customers are the most valuable, making recent purchases (0-30 days) with enormous spending (up to £275,000+)
# 
# 
# Cluster 0 (Orange) - High-Value customers are recent purchasers (0-50 days) with solid high spending (£10k-£70k range) and represent excellent growth opportunities
# 
# 
# Cluster 3 (Purple) - Regular customers have purchased recently (0-150 days) with low to moderate spending (under £15k) and form the stable customer base
# 
# 
# Cluster 1 (Green) - At-Risk/Lost customers haven't purchased in months (150-400+ days) with minimal spending and need immediate reactivation efforts.
# 
# 
# However, there's an exceptional case of Customer ID 17850 from the orange cluster (High-Value) who hasn't purchased in 302 days but has high frequency (35 transactions) and significant monetary value (£5,303.48), suggesting this is a previously loyal, valuable customer who may have temporarily stopped purchasing but retains high potential.

# #### 8.2.2 Frequency vs Monetary Analysis

# In[41]:


# 2. Frequency vs Monetary visualization
plt.figure(figsize=(12, 8))

for i in range(4):
    cluster_data = rfm[rfm['Cluster'] == i]
    plt.scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
               c=colors[i], label=f'Cluster {i}: {segment_names_short[i]}', 
               alpha=0.7, s=50)

plt.xlabel('Frequency (Number of Transactions)', fontsize=12)
plt.ylabel('Monetary (Total Spending £)', fontsize=12)
plt.title('Customer Segments: Frequency vs Monetary Value', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# **Interpretation of the graph:**
# 
# 
# This customer segmentation graph shows four distinct groups based on purchase frequency and spending
# 
# 
# Cluster 2 (Red) - VIP customers are the most valuable, containing customers who shop very frequently (50-250+ transactions) and spend enormous amounts (up to £275,000+)
# 
# 
# Cluster 0 (Orange) - High-Value customers shop regularly (10-80 transactions) with high spending (£10k-£70k) and are good targets for VIP upgrade programs
# 
# 
# Cluster 3 (Purple) - Regular customers represent typical customers who shop occasionally (1-20 transactions) with low spending and form the stable base
# 
# 
# Cluster 1 (Green) - At-Risk/Lost customers have minimal activity (1-5 transactions) with very low spending and need immediate attention.

# #### 8.2.3 Recency vs Frequency Analysis

# In[42]:


# 3. Recency vs Frequency visualization
plt.figure(figsize=(12, 8))

for i in range(4):
    cluster_data = rfm[rfm['Cluster'] == i]
    plt.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
               c=colors[i], label=f'Cluster {i}: {segment_names_short[i]}', 
               alpha=0.7, s=50)

plt.xlabel('Recency (Days Since Last Purchase)', fontsize=12)
plt.ylabel('Frequency (Number of Transactions)', fontsize=12)
plt.title('Customer Segments: Recency vs Frequency (Activity Level)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# **Interpretation of the graph:**
# 
# 
# This customer segmentation graph shows four distinct groups in order of priority:
# 
# 
# Cluster 2 (Red) - VIP customers are the most valuable, making recent purchases (0-20 days) with extremely high transaction frequency (up to 250 transactions), representing the most active and engaged customers
# 
# 
# Cluster 0 (Orange) - High-Value customers are recent purchasers (0-50 days) with moderate to high frequency (20-130 transactions) and represent excellent growth opportunities
# 
# 
# Cluster 3 (Purple) - Regular customers have purchased recently (0-100 days) with low frequency (1-20 transactions) and form the stable customer base
# 
# 
# Cluster 1 (Green) - At-Risk/Lost customers haven't purchased in months (150-400+ days) with minimal frequency and need immediate reactivation efforts.
# 
# 
# However, there's an exceptional case of an orange High-Value customer at 300 days recency with 35 transactions, suggesting a previously very active customer who has become dormant but retains high potential.

# #### 8.2.4 Customer Distribution by Segment

# In[43]:


# 4. Cluster sizes bar chart 
plt.figure(figsize=(12, 8))

cluster_counts = rfm['Cluster'].value_counts().sort_index()
cluster_percentages = (cluster_counts / len(rfm) * 100).round(1)

# Create bars with different colors for each cluster
bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on top of each bar 
for i, (count, percentage) in enumerate(zip(cluster_counts.values, cluster_percentages.values)):
    # Use different positioning for the tallest bar (Cluster 1)
    if i == 1:  # Cluster 1 (tallest bar)
        plt.text(i, count + 100, f'{count:,}\n({percentage}%)', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    else:
        plt.text(i, count + 50, f'{count:,}\n({percentage}%)', 
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

# Customize the chart
plt.xlabel('Customer Segments', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.title('Customer Distribution Across Segments', fontsize=14, fontweight='bold')

# Add cluster names to x-axis
plt.xticks(range(len(cluster_counts)), 
           [f'Cluster {i}\n{segment_names_short[i]}' for i in range(len(cluster_counts))])

plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(cluster_counts.values) * 1.15)  
plt.tight_layout()
plt.show()


# **Interpretation of the graph:**
# 
# This customer distribution chart reveals the composition of the customer base across segments
# 
# 
# Cluster 3 (Purple) - Regular customers dominates with 3,059 customers (70.9%), representing the majority of the customer base with typical low-value, occasional purchasing behavior
# 
# 
# Cluster 1 (Green) - At-Risk/Lost customers contains 1,054 customers (24.4%), indicating nearly a quarter of customers are dormant and need reactivation
# 
# 
# Cluster 0 (Orange) - High-Value customers has 193 customers (4.5%), representing a small but significant group of frequent, high-spending customers
# 
# 
# Cluster 2 (Red) - VIP customers contains only 11 customers (0.3%), but these ultra-high-value customers likely generate disproportionate revenue despite their tiny numbers.

# ## 9. Business Insights & Recommendations

# ### 9.1 Executive Summary

# In[44]:


total_customers = len(rfm)
total_revenue = df_clean['TotalAmount'].sum()

print(f"\n BUSINESS OVERVIEW:")
print(f"   • Total Customers Analyzed: {total_customers:,}")
print(f"   • Total Revenue: £{total_revenue:,.2f}")
print(f"   • Analysis Period: {(df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()).days} days")

print(f"\n CUSTOMER PORTFOLIO BREAKDOWN:")
for cluster in range(4):
    cluster_data = rfm[rfm['Cluster'] == cluster]
    cluster_customers = df_clean[df_clean['CustomerID'].isin(cluster_data.index)]
    cluster_revenue = cluster_customers['TotalAmount'].sum()
    revenue_percentage = (cluster_revenue / total_revenue * 100)
    
    segment_names = {
    0: "High Value Customers",
    1: "At-Rist/Lost Customers",
    2: "VIP Customers", 
    3: "Regular Customers"
    }
    
    print(f"   • {segment_names[cluster]}: {len(cluster_data):,} customers ({len(cluster_data)/total_customers*100:.1f}%)")
    print(f"     Revenue Contribution: £{cluster_revenue:,.2f} ({revenue_percentage:.1f}%)")

print(f"\n KEY FINDINGS:")
print(f"   • {rfm[rfm['Cluster']==0].shape[0]:,} customers (24.4%) are at risk of churning")
print(f"   • Only {rfm[rfm['Cluster']==2].shape[0]} customers (0.3%) are VIP Customers but drive significant revenue")
print(f"   • {rfm[rfm['Cluster']==1].shape[0]:,} customers (70.9%) form the core Regular customer base")
print(f"   • Opportunity to upgrade Regular customers to higher-value segments")


# ### 9.2 Business Recommendations by Segment

# **VIP CUSTOMERS (11 customers - 0.3%)**
# 
# 
# These customers generate the highest revenue and loyalty.
# 
# 
# Recommendations:
# 
# 
# • Provide dedicated customer service
# 
# • Offer exclusive products and early access
# 
# • Regular personal outreach and relationship management
# 
# • Premium loyalty rewards and benefits
# 
# **HIGH-VALUE CUSTOMERS (193 customers - 4.5%)**
# 
# 
# Strong customers with potential to become VIPs.
# 
# 
# Recommendations:
# 
# 
# • Cross-sell related products to increase order value
# 
# • Target for VIP program eligibility
# 
# • Send personalized product recommendations
# 
# • Re-engage dormant customer 17850 with special offer
# 
# **REGULAR CUSTOMERS (3,059 customers (70.9%))**
# 
# 
# Core customer base with room for growth.
# 
# 
# Recommendations:
# 
# 
# • Encourage more frequent purchases through email campaigns
# 
# • Offer bundle deals to increase transaction size
# 
# • Focus on upgrading top 20% to higher-value segments
# 
# • Implement loyalty program to build engagement
# 
# **AT-RISK/LOST 1,054 customers (24.4%)**
# 
# 
# Customers who may have churned, need immediate attention.
# 
# 
# Recommendations:
# 
# 
# • Launch win-back campaigns with discount offers
# 
# • Survey to understand why they stopped purchasing
# 
# • Use retargeting ads to re-engage
# 
# • If no response after 60 days, reduce marketing spend

# ### 9.3 Conclusion & Next Steps

# **PROJECT SUMMARY**
# 
# 
# This customer segmentation analysis successfully identified four distinct customer groups using RFM methodology, providing clear insights into customer behavior and business opportunities.
# 
# 
# **KEY BUSINESS INSIGHTS**
# 
# 
# • Customer base is heavily skewed toward Regular customers (71%)
# 
# • Small VIP segment (0.3%) likely drives disproportionate revenue
# 
# • Significant churn risk with 24% of customers inactive for 8+ months
# 
# • Clear upgrade path exists from Regular to Loyal High-Value segments
# 
# 
# **IMMEDIATE ACTION ITEMS**
# 
# 
# 1. Implement VIP retention program within 30 days
# 
# 
# 2. Launch win-back campaign for At-Risk customers
# 
# 
# 3. Create loyalty program to upgrade Regular customers
# 
# 
# 4. Develop targeted re-engagement for dormant high-value customers
# 
# 
# **RECOMMENDED FOLLOW-UP ANALYSIS**
# 
# 
# • Product affinity analysis to improve cross-selling
# 
# • Seasonal purchasing pattern analysis
# 
# • Customer lifetime value modeling
# 
# • Churn prediction modeling for early intervention
# 
# 
# **EXPECTED BUSINESS IMPACT**
# 
# 
# • Improved customer retention through targeted strategies
# 
# • Increased average order value via segment-specific campaigns
# 
# • More efficient marketing spend allocation
# 
# • Enhanced customer experience through personalized approaches

# In[ ]:




