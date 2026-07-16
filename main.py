import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create images folder
os.makedirs("images", exist_ok=True)

# Plot style
plt.style.use("ggplot")

print("=" * 50)
print("CodeAlpha - Task 2")
print("Unemployment Analysis with Python")
print("=" * 50)

# ==============================
# LOAD DATASETS
# ==============================

df1 = pd.read_excel("unemployment1.xlsx")
df2 = pd.read_excel("unemployment2.xlsx")

print("\n========== DATASET 1 ==========")
print(df1.head())

print("\n========== DATASET 2 ==========")
print(df2.head())

print("\n========== INFO DATASET 1 ==========")
print(df1.info())

print("\n========== INFO DATASET 2 ==========")
print(df2.info())

print("\n========== COLUMNS DATASET 1 ==========")
print(df1.columns.tolist())

print("\n========== COLUMNS DATASET 2 ==========")
print(df2.columns.tolist())

print("\n========== SHAPE ==========")
print("Dataset 1:", df1.shape)
print("Dataset 2:", df2.shape)

print("\n========== MISSING VALUES ==========")
print(df1.isnull().sum())
print()
print(df2.isnull().sum())

# ==============================
# DATA CLEANING
# ==============================

print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

# Remove spaces from column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Clean string columns
for col in df1.select_dtypes(include=["object", "string"]).columns:
    df1[col] = df1[col].astype(str).str.strip()

for col in df2.select_dtypes(include=["object","string"]).columns:
    df2[col] = df2[col].astype(str).str.strip()
print("\nUnique Frequency Values:")
print(df1["Frequency"].unique())
print("\nUpdated Column Names (Dataset 1)")
print(df1.columns.tolist())

print("\nUpdated Column Names (Dataset 2)")
print(df2.columns.tolist())

# Remove missing values
df1 = df1.dropna()
df2 = df2.dropna()

# Remove duplicate rows
df1 = df1.drop_duplicates()
df2 = df2.drop_duplicates()

print("\nDataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)

# Convert Date column
df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

print("\nDate column converted successfully!")

# ==============================
# EDA
# ==============================

print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset 1 Summary")
print(df1.describe())

print("\nDataset 2 Summary")
print(df2.describe())

print("\nNumber of Regions (Dataset 1):", df1["Region"].nunique())
print("Number of Regions (Dataset 2):", df2["Region"].nunique())

print("\nArea Distribution")
print(df1["Area"].value_counts())

print("\nFrequency Distribution")
print(df1["Frequency"].value_counts())

print("\nDataset 1 Date Range")
print(df1["Date"].min())
print(df1["Date"].max())

print("\nDataset 2 Date Range")
print(df2["Date"].min())
print(df2["Date"].max())

# ==============================
# VISUALIZATION 1
# ==============================

print("\nCreating Area Distribution Chart...")

plt.figure(figsize=(7,5))

sns.countplot(
    data=df1,
    x="Area"
)

plt.title("Urban vs Rural Records")
plt.xlabel("Area")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("images/area_distribution.png")

plt.show()

print("Area Distribution chart saved!")

print("\nProgram completed successfully!")
print("\nCreating Top 10 Regions Chart...")

top_regions = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
       .mean()
       .sort_values(ascending=False)
       .head(10)
)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_regions.values,
    y=top_regions.index
)

plt.title("Top 10 Regions by Average Unemployment Rate")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")

plt.tight_layout()

plt.savefig("images/top10_unemployment_regions.png")

plt.show()

print("Top 10 Regions chart saved!")
print("\nCreating Histogram...")

plt.figure(figsize=(8,5))

plt.hist(
    df1["Estimated Unemployment Rate (%)"],
    bins=20,
)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("images/histogram.png")

plt.show()

print("Histogram saved!")
print("\nCreating Boxplot...")

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df1["Estimated Unemployment Rate (%)"]
)

plt.title("Boxplot of Unemployment Rate")

plt.tight_layout()

plt.savefig("images/boxplot.png")

plt.show()

print("Boxplot saved!")
print("\nCreating Correlation Heatmap...")

plt.figure(figsize=(8,6))

numeric = df1.select_dtypes(include="number")

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("images/heatmap.png")

plt.show()

print("Heatmap saved!")
print("\nCreating Monthly Trend...")

monthly = (
    df1.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
)

plt.figure(figsize=(12,6))

plt.plot(
    monthly.index,
    monthly.values,
    marker="o"
)

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Average Rate (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/monthly_trend.png")

plt.show()

print("Monthly Trend saved!")
print("\n========== INSIGHTS ==========")

print(
    f"\nAverage Unemployment Rate: "
    f"{df1['Estimated Unemployment Rate (%)'].mean():.2f}%"
)

highest = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
       .mean()
       .idxmax()
)

highest_value = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
       .mean()
       .max()
)

print(
    f"Highest Average Unemployment Region: "
    f"{highest} ({highest_value:.2f}%)"
)

lowest = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
       .mean()
       .idxmin()
)

lowest_value = (
    df1.groupby("Region")["Estimated Unemployment Rate (%)"]
       .mean()
       .min()
)

print(
    f"Lowest Average Unemployment Region: "
    f"{lowest} ({lowest_value:.2f}%)"
)

print(
    f"\nHighest Recorded Rate: "
    f"{df1['Estimated Unemployment Rate (%)'].max()}%"
)

print(
    f"Lowest Recorded Rate: "
    f"{df1['Estimated Unemployment Rate (%)'].min()}%"
)