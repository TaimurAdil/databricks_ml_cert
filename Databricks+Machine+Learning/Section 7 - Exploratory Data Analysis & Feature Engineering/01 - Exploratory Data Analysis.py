# Databricks notebook source
# MAGIC %md
# MAGIC # 1) Explore the data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Libraries and Define the Schema

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Define the schema for the dataset

from pyspark.sql.types import DoubleType, StringType, StructType, StructField
 
schema = StructType([
  StructField("longitude", DoubleType(), True),
  StructField("latitude", DoubleType(), True),
  StructField("housing_median_age", DoubleType(), True),
  StructField("total_rooms", DoubleType(), True),
  StructField("total_bedrooms", DoubleType(), True),
  StructField("population", DoubleType(), True),
  StructField("households", DoubleType(), True),
  StructField("median_income", DoubleType(), True),
  StructField("median_house_value", DoubleType(), True),
  StructField("ocean_proximity", StringType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Data

# COMMAND ----------

df = spark.read.format("csv").schema(schema).option("header", "True").load("/FileStore/housing.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the results of DataFrame

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check the dataframe type

# COMMAND ----------

type(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert pyspark dataframe to pandas dataframe

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

type(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display the first few rows of the dataset

# COMMAND ----------

df.head()

# COMMAND ----------

df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine the dimensions

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Summary of a DataFrame

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Number of non-null values in each column

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve the column names

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine the data types of each column

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute the pairwise correlation of columns

# COMMAND ----------

df.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove duplicate rows

# COMMAND ----------

# Original shape of the df
original_shape = df.shape

# Drop duplicate rows
df = df.drop_duplicates()

# Calculate the number of duplicate rows
num_duplicate_rows = original_shape[0] - df.shape[0]

# print the number of duplicate rows removed
print("The number of duplicate rows removed :", num_duplicate_rows)

# COMMAND ----------

data = {
    'Name': ['John', 'Alice', 'John', 'Bob', 'Alice'],
    'Age': [25, 30, 25, 35, 30],
    'City': ['New York', 'London', 'New York', 'Paris', 'London'],
    'Age': [25, 30, 25, 35, 30],
    'City': ['New York', 'London', 'New York', 'Paris', 'London']
}

# COMMAND ----------

df_1 = pd.DataFrame(data)
print(df_1)

# COMMAND ----------

# Original shape of the df
original_shape = df_1.shape

# Drop duplicate rows
df_1 = df_1.drop_duplicates()

# Calculate the number of duplicate rows
num_duplicate_rows = original_shape[0] - df_1.shape[0]

# print the number of duplicate rows removed
print("The number of duplicate rows removed :", num_duplicate_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary statistics of numerical columns

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for missing values

# COMMAND ----------

# if there are any missing values in dataframe df
df.isnull().values.any()

# COMMAND ----------

# check how many missing values in df
df.isnull().values.sum()

# COMMAND ----------

# Number of missing values in each column
df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2) Visualize the Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation matrix and Heatmap

# COMMAND ----------

df.head()

# COMMAND ----------

df_2 = df.drop(columns="median_house_value")

# COMMAND ----------

df_2.head()

# COMMAND ----------

correlation_values = df_2.corrwith(df["median_house_value"])
sorted_correlation_values = correlation_values.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(
    sorted_correlation_values.index,
    sorted_correlation_values.values,
    color=['#1f77b4' if c > 0 else '#ff7f0e' for c in sorted_correlation_values.values]
)

ax.set_xlabel('Features')
ax.set_ylabel('Correlation')
ax.set_title('Correlation with Median House Value')

plt.xticks(rotation=45)
ax.grid(True)

plt.show()

# COMMAND ----------

corr = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title('Correlation Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Histogram of a numerical variable

# COMMAND ----------

plt.figure(figsize=(10, 6))  # Set the figure size

# Plot the histogram
plt.hist(df['median_house_value'], bins=20, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.title('Histogram of Median House Value')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scatter plot of two numerical variables

# COMMAND ----------

plt.figure(figsize=(10, 6))  # Set the figure size

# Plot the scatter plot
plt.scatter(df['median_house_value'], df['median_income'], alpha=0.5, color='skyblue', edgecolors='black')

# Add labels and title
plt.xlabel('Median House Value')
plt.ylabel('Median Income')
plt.title('Scatter Plot of Median House Value vs Median Income')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3) Pandas Profiling

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df,
                           correlations={
                               "auto": {"calculate": True},
                               "pearson": {"calculate": True},
                               "spearman": {"calculate": True},
                               "kendall": {"calculate": True},
                               "phi_k": {"calculate": True},
                               "cramers": {"calculate": True},
                           }, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)
