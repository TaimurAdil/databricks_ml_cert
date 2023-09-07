# Databricks notebook source
# MAGIC %md
# MAGIC ## Creating table for Regression

# COMMAND ----------

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
 
housing_df = spark.read.format("csv").schema(schema).option("header", "true").load("/FileStore/tables/housing.csv")

# COMMAND ----------

housing_df.write.saveAsTable("default.housing_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from housing_t

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating table for Classification

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField
 
schema = StructType([
  StructField("age", DoubleType(), True),
  StructField("workclass", StringType(), True),
  StructField("fnlwgt", DoubleType(), True),
  StructField("education", StringType(), True),
  StructField("education_num", DoubleType(), True),
  StructField("marital_status", StringType(), True),
  StructField("occupation", StringType(), True),
  StructField("relationship", StringType(), True),
  StructField("race", StringType(), True),
  StructField("sex", StringType(), True),
  StructField("capital_gain", DoubleType(), True),
  StructField("capital_loss", DoubleType(), True),
  StructField("hours_per_week", DoubleType(), True),
  StructField("native_country", StringType(), True),
  StructField("income", StringType(), True)
])
census_df = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")

# COMMAND ----------

census_df.write.saveAsTable("default.census_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM census_t

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating table for Forecasting

# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField, IntegerType
 
schema = StructType([
  StructField("date", StringType(), True),  
  StructField("county", StringType(), True),
  StructField("state", StringType(), True),
  StructField("fips", DoubleType(), True),
  StructField("cases", DoubleType(), True),
  StructField("deaths", DoubleType(), True)
])

covid_df = spark.read.format("csv").schema(schema).option("header", "true").load("/databricks-datasets/COVID/covid-19-data")


# COMMAND ----------

display(covid_df)

# COMMAND ----------

covid_df.write.saveAsTable("default.covid_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.covid_t

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from pyspark.sql.types import DoubleType, StringType, StructType, StructField, IntegerType
 
schema = StructType([
  StructField("date", StringType(), True),  
  StructField("county", StringType(), True),
  StructField("state", StringType(), True),
  StructField("fips", DoubleType(), True),
  StructField("cases", DoubleType(), True),
  StructField("deaths", DoubleType(), True)
])

covid_df = spark.read.format("csv").schema(schema).option("header", "true").load("/databricks-datasets/COVID/covid-19-data")

# COMMAND ----------

display(covid_df)

# COMMAND ----------

covid_df.dtypes

# COMMAND ----------

covid_df.write.saveAsTable("default.covid_t")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.covid_t

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table default.covid_t

# COMMAND ----------


