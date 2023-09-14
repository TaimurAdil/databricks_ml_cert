# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-a5b8d294-b5e1-4e6c-9d74-65b67c0d8762
# MAGIC %md 
# MAGIC
# MAGIC # Data Exploration and Transformation with *bamboolib*
# MAGIC
# MAGIC 8080 Labs was acquired in 2021 by Databricks. This brought the popular tool `bamboolib` into the Databricks ecosystem. As of the release of DBR 11.0. [bamboolib](https://docs.databricks.com/notebooks/bamboolib.html) is supported on Databricks clusters. Bamboolib is a **low-code** tool that accelerates Exploratory Data Analysis (EDA) and exploration with visual tools, as well as the ability to output code needed to perform the analysis after completion. 
# MAGIC
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC
# MAGIC  - Explore the Airbnb database of San Francisco using bamboolib
# MAGIC  - Use bamboolib to create a training dataset to predict prices based on relevant features
# MAGIC  - Build a Linear Regression model with Sci-Kit Learn to predict prices based on these features

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-dc0ee51e-e592-4b4b-8778-bcc2683b7339
# MAGIC %md 
# MAGIC
# MAGIC ## Initial Configuration
# MAGIC
# MAGIC Before we start, we need to install bamboolib and load the dataset that we will use.

# COMMAND ----------

# MAGIC %pip install bamboolib

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-bc703ccb-ad4c-400a-8a90-aecdbe05f9d8
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06-clean.delta/"
airbnb_pd = spark.read.format("delta").load(file_path).toPandas()

# COMMAND ----------

# DBTITLE 0,--i18n-0d11b16d-fb2e-45fb-9298-f2154a9d803e
# MAGIC %md 
# MAGIC
# MAGIC ## Explore the AirBnB dataset with bamboolib

# COMMAND ----------

# DBTITLE 0,--i18n-f5272d1d-5f8b-409d-b5b0-7fe28af8b685
# MAGIC %md 
# MAGIC
# MAGIC Now let's import pandas and bamboolib.

# COMMAND ----------

import pandas as pd
import bamboolib as bam

# COMMAND ----------

# DBTITLE 0,--i18n-1ad0e427-b70e-4ba4-867b-4b2b0a461e8c
# MAGIC %md 
# MAGIC
# MAGIC Bamboolib works directly with pandas dataframes, so we'll load the San Francisco Airbnb dataset and pull it into pandas for exploration.
# MAGIC
# MAGIC **💡 Question: if your dataset is too large to pull into a Pandas DataFrame directly, what should you do?**

# COMMAND ----------

# DBTITLE 0,--i18n-d2b967c7-c768-4ad9-9fc4-3ba23c4012c6
# MAGIC %md 
# MAGIC
# MAGIC Suppose we want to fit a **Linear Regression** to the `price` column. Let's use bamboolib to explore our dataset and find the right regressors to include in the model.
# MAGIC
# MAGIC
# MAGIC There are two main ways to access the bamboolib interface. The first is a button which appears above any printed Pandas DataFrame. 
# MAGIC
# MAGIC - Use the following cell to print `airbnb_pd` and click the green **"Show bamboolib UI"** button to access the bamboo GUI.
# MAGIC
# MAGIC - Click the **"Explore DataFrame"** button. Notice that we can see some summary values associated with each column of our DataFrame. 
# MAGIC
# MAGIC - We're interested in predicting the `price` column, so click on the row that corresponds to `price`.
# MAGIC
# MAGIC The **"Overview"** subtab for `price` will appear automatically with some summary statistics and a Histogram. You may notice the `price` values are heavily right skewed.
# MAGIC
# MAGIC - Click on the **"Predictors"** tab to see the the **Predictive Power Score (PPS)** for each variable regressed against `price`. The PPS ranges from 0 (low predictive power) to 1 (high predictive power). The highest Predictor in the dataset is `bedrooms` though only has a score of 0.139. Let's take a closer look at the relationship between `price` and `bedrooms`.
# MAGIC
# MAGIC - Click the **"Bivariate Plots"** subtab and select `price` and `bedrooms` as the plotting variables. 
# MAGIC
# MAGIC - Click on **"B predicts A"** and look at the Scatter plot. Here we can see a general linear trend that as the number of bedrooms increases the price of the Airbnb increases as well. However, we can also see there are some very expensive outliers! 
# MAGIC
# MAGIC We may be able to better fit the data by considering the `log(price)` rather than the `price` itself. Let's keep this plot here and move to the next cell to start manipulating our data.

# COMMAND ----------

airbnb_pd

# COMMAND ----------

# DBTITLE 0,--i18n-3b32e3d9-4bd6-4ec8-aa41-e4e6b6243966
# MAGIC %md 
# MAGIC
# MAGIC ### Create a new variable with bamboolib
# MAGIC
# MAGIC Let's look at how to add a new variable to our data as we start to clean and prepare our dataset for our machine learning model.
# MAGIC
# MAGIC - Print again the `airbnb_pd` DataFrame and open the bamboolib GUI.
# MAGIC
# MAGIC - In the "Search actions" box type "formula" and click on the option **"New Column Formula"** which will open a side-box popup. For the column name type `log_price` and for the column formula use `np.log(price)` and click the **"Execute"** button.
# MAGIC
# MAGIC - Click on the **"Explore DataFrame"** button and select the row associated with `log_price`. 
# MAGIC
# MAGIC - Click the **"Predictors"** subtab. Notice that the variable `accommodates` now has a PPS of 0.243, a major improvement over our best previous score of 0.139! You should see that `price` has almost a perfect PPS for `log_price`, that's expected. 
# MAGIC
# MAGIC ### Inspect correlation matrix
# MAGIC Other variables have better scores as well. There are high scores for `bedrooms`, `beds`, `room_type`, `host_total_listings_count`, and `bathrooms`, but might these variables be correlated with `accommodates`? If so, we shouldn't include all of them in our model because we would introduce multicollinearity. Let's investigate!
# MAGIC
# MAGIC - Go to the **"Exploration"** tab and select the **"Correlation Matrix"** subtab. 
# MAGIC
# MAGIC - In the **"Show"** box select the variables `accommodates`, `bedrooms`, `beds`, `host_total_listings_count`, and `bathrooms` (we can't select `room_type` for correlation as it is categorical), and click the Update button. 
# MAGIC
# MAGIC - Use your cursor to hover over the heatmap and examine the Pearson correlation of `accomodates` with each of the other variables. If we include `accommodates` in our model we should leave out `bedrooms` and `beds` as they have high correlation values above 0.75. However, the correlations with `bathrooms` and `host_total_listings_count` are pretty weak at 0.35 and -0.03, so these are safe to include in our model.
# MAGIC
# MAGIC ### Save final state
# MAGIC To save our progress for future users, let's use a feature of bamboolib to export our code. 
# MAGIC
# MAGIC - Go to the **"Data"** tab and look under the table of data to see a green **"Copy Code"** button. 
# MAGIC
# MAGIC - Click that and paste the code into the next cell.

# COMMAND ----------

airbnb_pd

# COMMAND ----------

#TODO : Fill in with copied code from above cell

# COMMAND ----------

# DBTITLE 0,--i18n-8dd8c24f-5e81-4c80-a36a-f8b5981947c8
# MAGIC %md 
# MAGIC
# MAGIC ## Prepare Dataset for Training the Linear Regresion Model

# COMMAND ----------

# DBTITLE 0,--i18n-17a314dc-cfa5-43b7-a935-93f4d55b76f2
# MAGIC %md 
# MAGIC
# MAGIC As we prepare to fit our linear regression, let's remove some columns that we don't need.
# MAGIC Use the below cell to execute `airbnb_pd` and open the bamboolib UI.
# MAGIC
# MAGIC ### Drop and transform columns
# MAGIC
# MAGIC One of the columns which was of interest in the last cell but was categorical was the `room_type` column. 
# MAGIC Use the **"Explore DataFrame"** button and view the details of this column. There are 3 distinct values: `Entire home/apt`, `Private room`, and `Shared room`. The value of this categorical variable could be encoded as an **Ordinal Variable** since they have a natural order of preference. Highest preference for most people would be to have additional privacy. Let's create a new column to represent this.
# MAGIC
# MAGIC - In the search actions box type "drop" and click **"Select or drop columns"**. 
# MAGIC
# MAGIC - Select **"Drop"** in the "Choose" pop up box and specify the following columns to drop: `bedrooms`, `beds`, and `price`. 
# MAGIC
# MAGIC - Specify `airbnb_reduced` as the new dataframe name. 
# MAGIC
# MAGIC - Click **"Explore DataFrame"**.
# MAGIC
# MAGIC - Under the **"Data"** tab select **"New Column Formula"** in the "Search actions box". Name your new variable `room_type_ord`.
# MAGIC
# MAGIC - For the formula we'll map the strings to integer values of increasing order with this formula; **`room_type.map({"Shared room" : 0, "Private room" : 1, "Entire home/apt" : 2})`**
# MAGIC
# MAGIC - Click **"Execute"**.
# MAGIC
# MAGIC ### Save final state
# MAGIC To save our progress for future users, let's use a feature of bamboolib to export our code. 
# MAGIC
# MAGIC - Go to the **"Data"** tab and look under the table of data to see a green **"Copy Code"** button. 
# MAGIC
# MAGIC - Click that and paste the code in the next cell.

# COMMAND ----------

airbnb_pd

# COMMAND ----------

#TODO : Fill in with copied code from above cell



# COMMAND ----------

# DBTITLE 0,--i18n-a66b3c10-5456-4c42-b3a7-e0a4deba0dc0
# MAGIC %md 
# MAGIC
# MAGIC ## Build training dataset
# MAGIC
# MAGIC Now that we have made some progress with bamboolib, let's take a subset of this data to build a training dataset to train our simple model. 
# MAGIC We'll use bamboolib to select the following columns for training: `accommodation`, `room_type_ord`, `host_total_listings_count`, and `bathrooms` to predict `log_price`.
# MAGIC
# MAGIC To build our training dataset:
# MAGIC - Print again the airbnb_pd DataFrame by running the next code cell. Open the bamboolib GUI.
# MAGIC
# MAGIC - In the search actions box type "drop" and click **"Select or drop columns"**.
# MAGIC
# MAGIC - Choose: **Select**.
# MAGIC
# MAGIC - Select following columns; `accommodates`, `room_type_ord`, `host_total_listings_count`, `bathrooms`, `log_price`.
# MAGIC
# MAGIC - Change the **New dataframe name** to **`airbnb_training`**.
# MAGIC
# MAGIC - Click **"Execute"** button.
# MAGIC
# MAGIC ### Save final state
# MAGIC To save our progress for future users, let's use a feature of bamboolib to export our code. 
# MAGIC
# MAGIC - Look under the table of data to see a green **"Copy Code"** button. 
# MAGIC
# MAGIC - Click that and paste the code in the next cell.

# COMMAND ----------

airbnb_reduced

# COMMAND ----------

#TODO : Fill in with copied code from above cell



# COMMAND ----------

# Answer
import pandas as pd; import numpy as np
# Step: Select columns
airbnb_training = airbnb_reduced[['accommodates', 'room_type_ord', 'host_total_listings_count', 'bathrooms', 'log_price']]

# COMMAND ----------

# DBTITLE 0,--i18n-344a7f9f-a551-4bcf-b42b-0ccffb85b72e
# MAGIC %md 
# MAGIC
# MAGIC ## Build a Simple Linear Regression Model to Predict Prices

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# We have our full dataset, let's split it into features and target
X = airbnb_training.copy()
y = X.pop('log_price')
# Let's split our data to training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# COMMAND ----------

# DBTITLE 0,--i18n-85a15c74-0736-4f3f-8e5c-4ed630db73c9
# MAGIC %md 
# MAGIC
# MAGIC Now that we have split our data, let's fit our model and test the model to see how accurate it is.

# COMMAND ----------

model = LinearRegression()
model = model.fit(X_train, y_train)

# Now we can test on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)

# COMMAND ----------

# DBTITLE 0,--i18n-a040e5c6-73b8-4d08-8e20-0f8930b1ac89
# MAGIC %md 
# MAGIC
# MAGIC Not bad! A mean squared error value of 0.24 for just a few input features is quite impressive. If we want to go further, we can add more features into the model. In the next lesson we will look at exploring this data with code.

# COMMAND ----------

# DBTITLE 0,--i18n-662cb315-e230-432f-b2b2-dc0fbbf170aa
# MAGIC %md 
# MAGIC
# MAGIC ## Classroom Cleanup
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
