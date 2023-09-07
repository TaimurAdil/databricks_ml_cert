# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction: Pandas Function APIs

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **There are three types of pandas function APIs:**
# MAGIC 1) [Grouped map](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html#pyspark-sql-groupeddata-applyinpandas)
# MAGIC 2) [Map](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.DataFrame.mapInPandas.html#pyspark-sql-dataframe-mapinpandas)
# MAGIC 3) [Cogrouped map](https://api-docs.databricks.com/python/pyspark/latest/pyspark.sql/api/pyspark.sql.PandasCogroupedOps.applyInPandas.html#pyspark-sql-pandascogroupedops-applyinpandas)

# COMMAND ----------

# MAGIC %md
# MAGIC # Grouped map

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You transform your grouped data using `groupBy().applyInPandas()` to implement the “split-apply-combine” pattern  
# MAGIC
# MAGIC **Split-apply-combine consists of three steps:**
# MAGIC
# MAGIC 1) Split the data into groups by using `DataFrame.groupBy`
# MAGIC 2) Apply a function on each group. The input and output of the function are both `pandas.DataFrame.` The input data contains all the rows and columns for each group
# MAGIC 3) Combine the results into a new `DataFrame` 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **To use `groupBy().applyInPandas()`, you must define the following:**
# MAGIC
# MAGIC 1) A Python function that defines the computation for each group
# MAGIC 2) A StructType object or a string that defines the schema of the output DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC ## Subtract the mean from each value in the group

# COMMAND ----------

# Creating a Spark DataFrame
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

# COMMAND ----------

df.show()

# COMMAND ----------

df.schema

# COMMAND ----------

# Defining a Custom Function (subtract_mean)
def subtract_mean(pdf): # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())

# COMMAND ----------

# Applying the Custom Function and Displaying the Resulting DataFrame
df.groupby("id").applyInPandas(subtract_mean, schema="id long, v double").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Map

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - You perform map operations with pandas instances by `DataFrame.mapInPandas()` in order to transform an iterator of `pandas.DataFrame` to another iterator of `pandas.DataFrame` that represents the current PySpark DataFrame and returns the result as a PySpark DataFrame
# MAGIC
# MAGIC - The underlying function takes and outputs an iterator of `pandas.DataFrame`
# MAGIC - It can return output of arbitrary length in contrast to some pandas UDFs such as Series to Series

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Transform an iterator of pandas.DataFrame to another iterator

# COMMAND ----------

# Creating a Spark DataFrame
df = spark.createDataFrame([(1, 21), (2, 30)], ("id", "age"))

# COMMAND ----------

df.show()

# COMMAND ----------

# Defining a Custom Filtering Function (filter_func)
def filter_func(iterator):
    for pdf in iterator:
        yield pdf[pdf.id == 1]

# COMMAND ----------

df.schema

# COMMAND ----------

# Applying the Custom Filtering Function using mapInPandas and Displaying the Resulting DataFrame
df.mapInPandas(filter_func, schema=df.schema).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cogrouped map

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - For cogrouped map operations with pandas instances, use `DataFrame.groupby().cogroup().applyInPandas()` to cogroup two PySpark DataFrames by a common key and then apply a Python function to each cogroup as shown:
# MAGIC
# MAGIC 1) Shuffle the data such that the groups of each DataFrame which share a key are cogrouped together
# MAGIC
# MAGIC 2) Apply a function to each cogroup. The input of the function is two pandas.DataFrame (with an optional tuple representing the key). The output of the function is a pandas.DataFrame
# MAGIC
# MAGIC 3) Combine the pandas.DataFrames from all groups into a new PySpark DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **To use groupBy().cogroup().applyInPandas(), you must define the following:**
# MAGIC 1) A Python function that defines the computation for each cogroup.
# MAGIC 2) A StructType object or a string that defines the schema of the output PySpark DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Perform an asof join between two datasets

# COMMAND ----------

import pandas as pd

df1 = spark.createDataFrame(
    [(20000101, 1, 1.0), (20000101, 2, 2.0), (20000102, 1, 3.0), (20000102, 2, 4.0)],
    ("time", "id", "v1"))

df2 = spark.createDataFrame(
    [(20000101, 1, "x"), (20000101, 2, "y")],
    ("time", "id", "v2"))

# COMMAND ----------

df1.show()

# COMMAND ----------

df2.show()

# COMMAND ----------

# Defining an "asof join" Function (asof_join)
def asof_join(l, r):
    return pd.merge_asof(l, r, on="time", by="id")

# COMMAND ----------

# Applying the "asof join" Function using cogroup and applyInPandas and Displaying the Resulting DataFrame
df1.groupby("id").cogroup(df2.groupby("id")).applyInPandas(
    asof_join, schema="time int, id int, v1 double, v2 string").show()

# COMMAND ----------


