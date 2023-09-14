# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-8c6d3ef3-e44b-4292-a0d3-1aaba0198525
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Data Cleansing
# MAGIC
# MAGIC We will be using Spark to do some exploratory data analysis & cleansing of the  San Francisco Airbnb rental dataset from <a href="http://insideairbnb.com/get-the-data.html" target="_blank">Inside Airbnb</a>.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/301/sf.jpg" style="height: 200px; margin: 10px; border: 1px solid #ddd; padding: 10px"/>
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Explore dataset based on summary statistics
# MAGIC * Identify and remove outliers in a dataset
# MAGIC * Impute missing data 
# MAGIC * Create an imputer pipeline using Spark ML

# COMMAND ----------

# DBTITLE 0,--i18n-1e2c921e-1125-4df3-b914-d74bf7a73ab5
# MAGIC %md 
# MAGIC ## 📌 Requirements
# MAGIC
# MAGIC **Required Databricks Runtime Version:** 
# MAGIC * Please note that in order to run this notebook, you must use one of the following Databricks Runtime(s): **12.2.x-cpu-ml-scala2.12**

# COMMAND ----------

# DBTITLE 0,--i18n-6a1bb996-7b50-4f03-9bcd-3d3ec3069a6d
# MAGIC %md 
# MAGIC ## Lesson Setup
# MAGIC
# MAGIC The first thing we're going to do is to **run setup script**. This script will define the required configuration variables that are scoped to each user.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# DBTITLE 0,--i18n-969507ea-bffc-4255-9a99-2306a594625f
# MAGIC %md 
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Let's load the Airbnb dataset in.

# COMMAND ----------

file_path = f"{DA.paths.datasets}/airbnb/sf-listings/sf-listings-2019-03-06.csv"

raw_df = spark.read.csv(file_path, header="true", inferSchema="true", multiLine="true", escape='"')

display(raw_df)

# COMMAND ----------

raw_df.columns

# COMMAND ----------

# DBTITLE 0,--i18n-94856418-c319-4915-a73e-5728fcd44101
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC For the sake of simplicity, only keep certain columns from this dataset. We will talk about feature selection later.

# COMMAND ----------

columns_to_keep = [
    "host_is_superhost",
    "cancellation_policy",
    "instant_bookable",
    "host_total_listings_count",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "bed_type",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "price"
]

base_df = raw_df.select(columns_to_keep)
base_df.cache().count()
display(base_df)

# COMMAND ----------

# DBTITLE 0,--i18n-a12c5a59-ad1c-4542-8695-d822ec10c4ca
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC  
# MAGIC ## Fixing Data Types
# MAGIC
# MAGIC Take a look at the schema above. You'll notice that the **`price`** field got picked up as string. For our task, we need it to be a numeric (double type) field. 
# MAGIC
# MAGIC Let's fix that.

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixed_price_df = base_df.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4ad08138-4563-4a93-b038-801832c9bc73
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Summary statistics
# MAGIC
# MAGIC Two options:
# MAGIC * **`describe`**: count, mean, stddev, min, max
# MAGIC * **`summary`**: describe + interquartile range (IQR)
# MAGIC
# MAGIC **Question:** When to use IQR/median over mean? Vice versa?

# COMMAND ----------

display(fixed_price_df.describe())

# COMMAND ----------

display(fixed_price_df.summary())

# COMMAND ----------

# DBTITLE 0,--i18n-bd55efda-86d0-4584-a6fc-ef4f221b2872
# MAGIC %md 
# MAGIC
# MAGIC ### Explore Dataset with Data Profile
# MAGIC
# MAGIC The **Data Profile** feature in Databricks notebooks offers valuable insights and benefits for data analysis and exploration. By leveraging Data Profile, users gain a comprehensive overview of their **dataset's characteristics, statistics, and data quality metrics**. This feature enables data scientists and analysts to understand the data distribution, identify missing values, detect outliers, and explore descriptive statistics efficiently.
# MAGIC
# MAGIC There are two ways of viewing Data Profiler. The first option is the UI.
# MAGIC
# MAGIC - After using `display` function to show a data frame, click **+** icon next to the *Table* in the header. 
# MAGIC - Click **Data Profile**. 
# MAGIC
# MAGIC
# MAGIC
# MAGIC This functionality is also available through the dbutils API in Python, Scala, and R, using the dbutils.data.summarize(df) command. We can also use **`dbutils.data.summarize(df)`** to display Data Profile UI.
# MAGIC
# MAGIC Note that this features will profile the entire data set in the data frame or SQL query results, not just the portion displayed in the table

# COMMAND ----------

dbutils.data.summarize(fixed_price_df)

# COMMAND ----------

# DBTITLE 0,--i18n-e9860f92-2fbe-4d23-b728-678a7bb4734e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Getting rid of extreme values
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the **`price`** column.

# COMMAND ----------

display(fixed_price_df.select("price").describe())

# COMMAND ----------

# DBTITLE 0,--i18n-4a8fe21b-1dac-4edf-a0a3-204f170b05c9
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC There are some super-expensive listings, but it's up to the SME (Subject Matter Experts) to decide what to do with them. We can certainly filter the "free" Airbnbs though.
# MAGIC
# MAGIC Let's see first how many listings we can find where the *price* is zero.

# COMMAND ----------

fixed_price_df.filter(col("price") == 0).count()

# COMMAND ----------

# DBTITLE 0,--i18n-bf195d9b-ea4d-4a3e-8b61-372be8eec327
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Now only keep rows with a strictly positive *price*.

# COMMAND ----------

pos_prices_df = fixed_price_df.filter(col("price") > 0)

# COMMAND ----------

# DBTITLE 0,--i18n-dc8600db-ebd1-4110-bfb1-ce555bc95245
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's take a look at the *min* and *max* values of the *minimum_nights* column:

# COMMAND ----------

display(pos_prices_df.select("minimum_nights").describe())

# COMMAND ----------

display(pos_prices_df
        .groupBy("minimum_nights").count()
        .orderBy(col("count").desc(), col("minimum_nights"))
       )

# COMMAND ----------

# DBTITLE 0,--i18n-5aa4dfa8-d9a1-42e2-9060-a5dcc3513a0d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC A minimum stay of one year seems to be a reasonable limit here. Let's filter out those records where the *minimum_nights* is greater than 365.

# COMMAND ----------

min_nights_df = pos_prices_df.filter(col("minimum_nights") <= 365)

display(min_nights_df)

# COMMAND ----------

# DBTITLE 0,--i18n-25a35390-d716-43ad-8f51-7e7690e1c913
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Handling Null Values
# MAGIC
# MAGIC There are a lot of different ways to handle null values. Sometimes, null can actually be a key indicator of the thing you are trying to predict (e.g. if you don't fill in certain portions of a form, probability of it getting approved decreases).
# MAGIC
# MAGIC Some ways to handle nulls:
# MAGIC * Drop any records that contain nulls
# MAGIC * Numeric:
# MAGIC   * Replace them with mean/median/zero/etc.
# MAGIC * Categorical:
# MAGIC   * Replace them with the mode
# MAGIC   * Create a special category for null
# MAGIC * Use techniques like ALS (Alternating Least Squares) which are designed to impute missing values
# MAGIC   
# MAGIC **If you do ANY imputation techniques for categorical/numerical features, you MUST include an additional field specifying that field was imputed.**
# MAGIC
# MAGIC SparkML's Imputer (covered below) does not support imputation for categorical features.

# COMMAND ----------

# DBTITLE 0,--i18n-83e56fca-ce6d-4e3c-8042-0c1c7b9eaa5a
# MAGIC %md 
# MAGIC
# MAGIC ### Impute: Cast to Double
# MAGIC
# MAGIC Imputing in the context of data means replacing missing values with something intentional, such as replacing nulls with an average/mean value. SparkML's <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html?highlight=imputer#pyspark.ml.feature.Imputer" target="_blank">Imputer </a> requires all fields to be of type double. Let's cast all integer fields to double.

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integer_columns = [x.name for x in min_nights_df.schema.fields if x.dataType == IntegerType()]
doubles_df = min_nights_df

for c in integer_columns:
    doubles_df = doubles_df.withColumn(c, col(c).cast("double"))

columns = "\n - ".join(integer_columns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

# DBTITLE 0,--i18n-69b58107-82ad-4cec-8984-028a5df1b69e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC Add a dummy column to denote the presence of null values before imputing (i.e. 1.0 = Yes, 0.0 = No).

# COMMAND ----------

from pyspark.sql.functions import when

impute_cols = [
    "bedrooms",
    "bathrooms",
    "beds", 
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value"
]

for c in impute_cols:
    doubles_df = doubles_df.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doubles_df.describe())

# COMMAND ----------

# DBTITLE 0,--i18n-c88f432d-1252-4acc-8c91-4834c00da789
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Transformers and Estimators
# MAGIC
# MAGIC Spark ML standardizes APIs for machine learning algorithms to make it easier to combine multiple algorithms into a single pipeline, or workflow. Let's cover two key concepts introduced by the Spark ML API: **`transformers`** and **`estimators`**.
# MAGIC
# MAGIC **Transformer**: Transforms one DataFrame into another DataFrame. It accepts a DataFrame as input, and returns a new DataFrame with one or more columns appended to it. Transformers do not learn any parameters from your data and simply apply rule-based transformations. It has a **`.transform()`** method.
# MAGIC
# MAGIC **Estimator**: An algorithm which can be fit on a DataFrame to produce a Transformer. E.g., a learning algorithm is an Estimator which trains on a DataFrame and produces a model. It has a **`.fit()`** method because it learns (or "fits") parameters from your DataFrame.

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=impute_cols, outputCols=impute_cols)

imputer_model = imputer.fit(doubles_df)
imputed_df = imputer_model.transform(doubles_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4df06e83-27e6-4cc6-b66d-883317b2a7eb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC OK, our data is cleansed now. Let's save this DataFrame to Delta so that we can start building models with it.

# COMMAND ----------

imputed_df.write.format("delta").mode("overwrite").save(f"{DA.paths.working_dir}/imputed_results")

# COMMAND ----------

# DBTITLE 0,--i18n-a2c7fb12-fd0b-493f-be4f-793d0a61695b
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
