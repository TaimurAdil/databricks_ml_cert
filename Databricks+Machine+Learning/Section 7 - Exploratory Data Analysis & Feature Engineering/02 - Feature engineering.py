# Databricks notebook source
# MAGIC %md
# MAGIC # Load the dataset

# COMMAND ----------

import pandas as pd
import numpy as np

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

# Load the dataset
df = spark.read.format("csv").schema(schema).option("header", "True").load("/FileStore/housing.csv")

# COMMAND ----------

# Convert pyspark dataframe to pandas dataframe
df = df.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1) Missing value imputation

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.fillna(df.mean(numeric_only=True), inplace=True)

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

df.select_dtypes(include="object").columns

# COMMAND ----------

df['ocean_proximity'] = df['ocean_proximity'].fillna(df['ocean_proximity'].mode()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC # 2) Outlier removal

# COMMAND ----------

# MAGIC %md
# MAGIC **Outlier removal** is a process of identifying and removing or modifying data points that are considered unusual or extreme compared to the majority of the dataset

# COMMAND ----------

# MAGIC %md
# MAGIC **There are several methods commonly used to remove outliers from a DataFrame. Here are a few of them:**
# MAGIC
# MAGIC **1) Z-Score Method:**
# MAGIC - Calculate the z-score for each value in the DataFrame.
# MAGIC - Remove rows where any column has a z-score greater than a predefined threshold (e.g., 3).
# MAGIC - This method assumes that the data follows a normal distribution.
# MAGIC
# MAGIC **2) IQR (Interquartile Range) Method:**
# MAGIC - Calculate the IQR for each column in the DataFrame.
# MAGIC - Remove rows where any column value is below the first quartile minus a multiple of the IQR or above the third quartile plus a multiple of the IQR (e.g., 1.5 times the IQR).
# MAGIC - This method is robust to non-normal distributions.
# MAGIC
# MAGIC **3) Tukey's Fences Method:**
# MAGIC - Calculate the lower and upper fences based on the first and third quartiles and the IQR.
# MAGIC - Remove rows where any column value is below the lower fence or above the upper fence (e.g., 1.5 times the IQR).
# MAGIC - Similar to the IQR method, this approach is robust to non-normal distributions.
# MAGIC
# MAGIC **4) Standard Deviation Method:**
# MAGIC - Calculate the mean and standard deviation for each column in the DataFrame.
# MAGIC - Remove rows where any column value is above or below a certain number of standard deviations from the mean (e.g., 3 standard deviations).
# MAGIC - This method assumes a normal distribution of the data.
# MAGIC
# MAGIC **5) Percentile Method:**
# MAGIC - Calculate the lower and upper percentiles for each column in the DataFrame (e.g., 1st and 99th percentiles).
# MAGIC - Remove rows where any column value is below the lower percentile or above the upper percentile.
# MAGIC - This method is not distribution-specific and removes extreme values.
# MAGIC
# MAGIC
# MAGIC It's important to note that the choice of method depends on the characteristics of your data and the specific requirements of your analysis. You may need to experiment with different methods or use a combination of approaches to effectively remove outliers from your DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ## IQR (Interquartile Range) Method

# COMMAND ----------

df.dtypes

# COMMAND ----------

# Define the numerical columns
numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                     'total_bedrooms', 'population', 'households', 'median_income',
                     'median_house_value']

# Calculate the first quartile (Q1) and third quartile (Q3) for each numerical column
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)

# Calculate the interquartile range (IQR) for each numerical column
IQR = Q3 - Q1

# Define the lower and upper bounds for outlier detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers from the DataFrame
df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]

# COMMAND ----------

df_no_outliers.shape

# COMMAND ----------

# Calculate the initial row count
initial_row_count = len(df)

# Calculate the row count after outlier removal
final_row_count = len(df_no_outliers)

# Calculate the number of removed rows
removed_rows = initial_row_count - final_row_count

# Display the number of removed rows
print("Number of removed rows:", removed_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Z-Score Method

# COMMAND ----------

from scipy import stats

# Define the numerical columns
numerical_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                     'total_bedrooms', 'population', 'households', 'median_income',
                     'median_house_value']

# Calculate z-scores for numerical columns
z_scores = stats.zscore(df[numerical_columns])

# Define the threshold for outlier detection
threshold = 3

# Filter out the outliers from the DataFrame
df_no_outliers = df[(z_scores < threshold).all(axis=1)]

# COMMAND ----------

df_no_outliers.shape

# COMMAND ----------

# Calculate the initial row count
initial_row_count = len(df)

# Calculate the row count after outlier removal
final_row_count = len(df_no_outliers)

# Calculate the number of removed rows
removed_rows = initial_row_count - final_row_count

# Display the number of removed rows
print("Number of removed rows:", removed_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3) Feature Creation

# COMMAND ----------

df_no_outliers.shape

# COMMAND ----------

df.shape

# COMMAND ----------

df = df_no_outliers

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df["housing_median_age_days"] = df["housing_median_age"] * 365

# COMMAND ----------

df.head()

# COMMAND ----------

df = df.drop(columns="housing_median_age_days")

# COMMAND ----------

df.head()

# COMMAND ----------

# Create a new feature by adding existing features
df['new_feature'] = df['feature1'] + df['feature2']

# COMMAND ----------

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
polynomial_features = poly.fit_transform(df[['feature1', 'feature2']])

# Interaction features
df['interaction_feature'] = df['feature1'] * df['feature2']

# Binning/Discretization
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['child', 'young', 'middle-aged', 'elderly'])

# Encoding categorical variables
encoded_df = pd.get_dummies(df, columns=['category'], prefix='category', drop_first=True)

# Textual feature extraction (using CountVectorizer for bag-of-words)
from sklearn.feature_extraction.text import CountVectorizer

text_data = ['This is the first document.', 'This document is the second document.']
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(text_data).toarray()

# Time-based features
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['month'] = pd.to_datetime(df['date']).dt.month

# Domain-specific feature creation
df['income_ratio'] = df['income'] / df['expenses']
df['aggregate_feature'] = df['feature1'] + df['feature2']

# COMMAND ----------

# MAGIC %md
# MAGIC # 4) Feature Scaling

# COMMAND ----------

# MAGIC %md
# MAGIC - **Feature scaling, also known as data normalization:** The process of transforming numerical features in a dataset to a common scale. It is a crucial step in data preprocessing and feature engineering, as it helps to bring the features to a similar range and magnitude. The goal of feature scaling is to ensure that no single feature dominates the learning process or introduces bias due to its larger values
# MAGIC
# MAGIC - **There are two common methods for feature scaling:**
# MAGIC
# MAGIC **1) Standardization (Z-score normalization):** In this method, each feature is transformed to have zero mean and unit variance. The formula for standardization is: x_scaled = (x - mean) / standard_deviation.
# MAGIC Standardization ensures that the transformed feature has a mean of 0 and a standard deviation of 1.
# MAGIC
# MAGIC **2) Min-Max scaling:** In this method, each feature is scaled to a specific range, typically between 0 and 1.
# MAGIC The formula for min-max scaling is: x_scaled = (x - min) / (max - min).
# MAGIC Min-max scaling preserves the relative ordering of values and ensures that the transformed feature is bounded within the defined range.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Feature scaling is important for several reasons:**
# MAGIC
# MAGIC 1) Gradient-based optimization algorithms, such as gradient descent, converge faster when features are on a similar scale. This helps in achieving faster convergence and more efficient training of machine learning models.
# MAGIC
# MAGIC 2) Features with larger scales can dominate the learning process, leading to biased results. Scaling the features ensures that no single feature has undue influence on the model.
# MAGIC
# MAGIC 3) Many machine learning algorithms, such as K-nearest neighbors (KNN) and support vector machines (SVM), rely on calculating distances between data points. If features are not on a similar scale, features with larger values can dominate the distance calculations, leading to suboptimal results.
# MAGIC
# MAGIC 4) Some algorithms, such as principal component analysis (PCA), assume that the data is centered and on a similar scale. Feature scaling is necessary to meet these assumptions and obtain meaningful results.

# COMMAND ----------

df.columns

# COMMAND ----------

numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

# COMMAND ----------

print(numerical_features)

# COMMAND ----------

df.head()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5) One-hot-encoding (Feature Encoding)

# COMMAND ----------

# MAGIC %md
# MAGIC One-hot encoding, also known as feature encoding, is a technique used to convert categorical variables into a numerical representation that can be used by machine learning algorithms. It is a common preprocessing step in machine learning tasks that involve categorical features.
# MAGIC
# MAGIC Categorical variables are variables that represent qualitative or discrete characteristics or groups. Examples of categorical variables include "color" (red, green, blue), "city" (New York, London, Paris), or "animal" (cat, dog, bird).

# COMMAND ----------

# MAGIC %md
# MAGIC **The benefits of one-hot encoding include:**
# MAGIC
# MAGIC 1) Compatibility with machine learning algorithms: Many machine learning algorithms require numerical input. By converting categorical variables into a numerical format, one-hot encoding enables the use of these variables in machine learning models.
# MAGIC
# MAGIC 2) Preserving information: One-hot encoding preserves the information about the presence or absence of specific categories in the original data, which can be valuable for certain models.

# COMMAND ----------

df.select_dtypes(include="object").columns

# COMMAND ----------

df['ocean_proximity'].unique()

# COMMAND ----------

df['ocean_proximity'].nunique()

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

df = pd.get_dummies(data=df, drop_first=True)

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # 6) Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Feature selection is a crucial step in feature engineering, where the goal is to identify and select a subset of relevant features from the available set of features in a dataset. The aim is to improve model performance, reduce overfitting, enhance interpretability, and reduce computational complexity.
# MAGIC
# MAGIC - **Benefits of feature selection include:**
# MAGIC
# MAGIC 1) Improved model performance: By selecting relevant features, feature selection can enhance model accuracy, reduce overfitting, and improve generalization on unseen data.
# MAGIC
# MAGIC 2) Faster model training: Fewer features can lead to faster training times, especially when dealing with large datasets or complex models.
# MAGIC
# MAGIC 3) Enhanced interpretability: Selecting a subset of meaningful features can improve the interpretability of the model, allowing for better understanding and insights.
# MAGIC
# MAGIC 4) Reduced dimensionality: By eliminating irrelevant or redundant features, feature selection can reduce the dimensionality of the dataset, making it more manageable and reducing the risk of the curse of dimensionality.

# COMMAND ----------

df.head()

# COMMAND ----------

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# COMMAND ----------

X.head()

# COMMAND ----------

y.head()

# COMMAND ----------

type(y)

# COMMAND ----------

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Create the feature selection model (linear regression estimator and 5 features to select)
estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=5)

# Fit the feature selection model on the data
rfe.fit(X, y)

# Get the selected features
selected_features = X.columns[rfe.support_]
print(selected_features)

# COMMAND ----------

# MAGIC %md
# MAGIC # 7) Feature Transformation (if needed)

# COMMAND ----------

# MAGIC %md
# MAGIC - The process of applying mathematical or statistical transformations to the existing features in a dataset to make them more suitable for a machine learning algorithm or to reveal underlying patterns in the data.
# MAGIC - Feature transformation techniques aim to improve the quality and representativeness of the features, which can lead to better model performance and more meaningful insights.

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import boxcox

# Example dataset
data = pd.DataFrame({
    'feature1': [10, 20, 30, 40, 50],
    'feature2': [0.1, 1, 10, 100, 1000],
    'feature3': [100, 200, 300, 400, 500]
})

# COMMAND ----------

print(data)

# COMMAND ----------

# Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)

# Logarithmic Transformation
log_transformed_feature = np.log(data['feature1'])

# Power Transformation
power_transformed_feature = np.sqrt(data['feature2'])

# Box-Cox Transformation
boxcox_transformed_feature, _ = boxcox(data['feature3'])

# Binning
bin_edges = [0, 20, 40, 60]
binned_feature = pd.cut(data['feature1'], bins=bin_edges, labels=False)

# Polynomial Transformation
polynomial_features = pd.DataFrame({
    'feature1_squared': data['feature1'] ** 2,
    'feature1_cubed': data['feature1'] ** 3
})

# Interaction Terms
interaction_feature = data['feature1'] * data['feature2']

# COMMAND ----------

# Print the transformed features
print("Normalized data:")
print(normalized_data)

print("\nStandardized data:")
print(standardized_data)

print("\nLogarithmic transformed feature:")
print(log_transformed_feature)

print("\nPower transformed feature:")
print(power_transformed_feature)

print("\nBox-Cox transformed feature:")
print(boxcox_transformed_feature)

print("\nBinned feature:")
print(binned_feature)

print("\nPolynomial features:")
print(polynomial_features)

print("\nInteraction feature:")
print(interaction_feature)

# COMMAND ----------

# MAGIC %md
# MAGIC # 8) Dimensionality Reduction (if needed)

# COMMAND ----------

# MAGIC %md
# MAGIC - The process of reducing the number of features or variables in a dataset while preserving the essential information
# MAGIC - **Aims to overcome,**
# MAGIC 1) The curse of dimensionality
# MAGIC 2) Improve computational efficiency
# MAGIC 3) Eliminate noise or redundant features
# MAGIC 4) Potentially enhance the performance of ML models
# MAGIC
# MAGIC - High-dimensional data can lead to several challenges, such as increased computational complexity, overfitting, and difficulty in interpreting and visualizing the data
# MAGIC - Dimensionality reduction techniques address these challenges by transforming or projecting the data into a lower-dimensional space, where the most relevant information is retained.

# COMMAND ----------

# MAGIC %md
# MAGIC **There are two main approaches to dimensionality reduction:**
# MAGIC
# MAGIC **1) Feature Selection:** This approach involves selecting a subset of the original features based on certain criteria. It aims to identify the most informative and relevant features that contribute significantly to the target variable or capture the underlying patterns in the data. Feature selection methods can be filter-based (e.g., correlation, statistical tests) or wrapper-based (e.g., recursive feature elimination, forward/backward feature selection).
# MAGIC
# MAGIC **2) Feature Extraction:** This approach involves transforming the original features into a new set of lower-dimensional features. It aims to create a compressed representation of the data by combining or projecting the original features into a new feature space. Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) are popular feature extraction techniques. Other methods include Non-negative Matrix Factorization (NMF), t-SNE, and Autoencoders.

# COMMAND ----------

# MAGIC %md
# MAGIC **Benefits of dimensionality reduction:**
# MAGIC
# MAGIC 1) Computational Efficiency: By reducing the number of features, the computational complexity of algorithms decreases, resulting in faster training and inference times.
# MAGIC
# MAGIC 2) Overfitting Prevention: Dimensionality reduction helps to remove noisy or irrelevant features, reducing the risk of overfitting and improving the generalization capability of models.
# MAGIC
# MAGIC 3) Improved Visualization: Lower-dimensional data can be visualized more easily, enabling better understanding and interpretation of the data.
# MAGIC
# MAGIC 4) Enhanced Model Performance: By focusing on the most relevant features, dimensionality reduction can improve the performance of machine learning models by reducing noise, capturing important patterns, and avoiding the curse of dimensionality.

# COMMAND ----------

# MAGIC %md
# MAGIC **Principal Component Analysis (PCA)** is a widely used technique for dimensionality reduction and feature extraction in data analysis and machine learning. It aims to transform a high-dimensional dataset into a lower-dimensional space while preserving the most important patterns and variations in the data

# COMMAND ----------

from sklearn.decomposition import PCA

# Create the PCA model
pca = PCA(n_components=2)

# Fit the PCA model to X
pca.fit(X)

# Transform X to the new feature space
X_reduced = pca.transform(X)

# Print the shape of X_reduced
print(X_reduced.shape)

# COMMAND ----------

# Print the number of principal components
print(pca.n_components_)

# COMMAND ----------

print(X_reduced)
