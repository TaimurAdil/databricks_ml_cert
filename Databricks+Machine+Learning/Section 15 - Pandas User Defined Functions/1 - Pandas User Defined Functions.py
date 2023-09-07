# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - **Pandas user-defined function (UDF):**  also known as vectorized UDF
# MAGIC - It is a user-defined function that uses Apache Arrow to transfer data and pandas to work with the data
# MAGIC - Pandas UDFs allow vectorized operations that can increase performance up to 100x compared to row-at-a-time Python UDFs

# COMMAND ----------

# MAGIC %md
# MAGIC # Type hint in pandas udf

# COMMAND ----------

# MAGIC %md
# MAGIC - In Python, type hints are used to statically indicate the type of a value. This can be done for variables, parameters, function arguments, and return values
# MAGIC - In the context of Pandas UDFs, type hints are used to specify the types of the input and output arguments of the UDF
# MAGIC - This can help to improve the **the readability, maintainability of the code, and helps to catch errors at compile time**

# COMMAND ----------

import pandas as pd

# COMMAND ----------

def multiply(a: pd.Series, b: pd.Series) -> pd.Series:
    return a*b

# COMMAND ----------

x = pd.Series([2, 3, 4])
y = pd.Series([5, 6, 7])

# COMMAND ----------

print(multiply(x, y))

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Series to Series UDF

# COMMAND ----------

# MAGIC %md
# MAGIC - **Use:** To vectorize scalar operations & with APIs such as `select` and `withColumn`
# MAGIC - Spark runs a pandas UDF by splitting columns into batches, calling the function for each batch as a subset of the data, then concatenating the results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a pandas UDF that computes the product of 2 columns

# COMMAND ----------

# Import necessary libraries
import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# COMMAND ----------

# Declare the function and create the UDF
def multiply_func(a: pd.Series, b: pd.Series) -> pd.Series:
    return a * b

multiply = pandas_udf(multiply_func, returnType=LongType())

# COMMAND ----------

# Test the Function Locally
x = pd.Series([1, 2, 3])
print(multiply_func(x, x))

# COMMAND ----------

# Create a Spark DataFrame
df = spark.createDataFrame(pd.DataFrame(x, columns=["x"]))

# COMMAND ----------

df.show()

# COMMAND ----------

# Execute function as a Spark vectorized UDF
df.select(multiply(col("x"), col("x"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Iterator of Series to Iterator of Series UDF

# COMMAND ----------

# MAGIC %md
# MAGIC **An iterator UDF is the same as a scalar pandas UDF except:**
# MAGIC 1) The Python function: Takes an iterator of batches instead of a single input batch as input, Returns an iterator of output batches instead of a single output batch
# MAGIC 2) The length of the entire output in the iterator should be the same as the length of the entire input
# MAGIC 3) The wrapped pandas UDF takes a single Spark column as an input

# COMMAND ----------

# MAGIC %md
# MAGIC - Useful when the UDF execution requires initializing some state
# MAGIC - **For example,** loading a ML model file to apply inference to every input batch

# COMMAND ----------

# Import necessary libraries
import pandas as pd
from typing import Iterator
from pyspark.sql.functions import col, pandas_udf, struct

# COMMAND ----------

# Create a Pandas DataFrame
pdf = pd.DataFrame([1, 2, 3], columns=["x"])

# Convert the Pandas DataFrame to a Spark DataFrame
df = spark.createDataFrame(pdf)

# COMMAND ----------

df.show()

# COMMAND ----------

# Define a Pandas UDF 'plus_one'

# When the UDF is called with the column, the input to the underlying function is an iterator of pd.Series
@pandas_udf("long")
def plus_one(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    for x in batch_iter:
        yield x + 1

# COMMAND ----------

# Apply the 'plus_one' UDF to the "x" column of the Spark DataFrame
df.select(plus_one(col("x"))).show()

# COMMAND ----------

# Define another Pandas UDF named 'plus_y'

y_bc = spark.sparkContext.broadcast(1)

@pandas_udf("long")
def plus_y(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    y = y_bc.value  # initialize states
    try:
        for x in batch_iter:
            yield x + y
    finally:
        pass  # release resources here, if any

# COMMAND ----------

# Apply the 'plus_y' UDF to the "x" column of the Spark DataFrame
df.select(plus_y(col("x"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Iterator of multiple Series to Iterator of Series UDF

# COMMAND ----------

# MAGIC %md
# MAGIC - Similar characteristics and restrictions as Iterator of Series to Iterator of Series UDF
# MAGIC - The specified function takes an iterator of batches and outputs an iterator of batches
# MAGIC - It is also useful when the UDF execution requires initializing some state

# COMMAND ----------

# MAGIC %md
# MAGIC **The differences are:**
# MAGIC - The underlying Python function takes an iterator of a tuple of pandas Series
# MAGIC - The wrapped pandas UDF takes multiple Spark columns as an input

# COMMAND ----------

# Importing Libraries
import pandas as pd
from typing import Iterator, Tuple
from pyspark.sql.functions import col, pandas_udf, struct

# COMMAND ----------

# Create a Pandas DataFrame
pdf = pd.DataFrame([1, 2, 3], columns=["x"])

# Convert the Pandas DataFrame to a Spark DataFrame
df = spark.createDataFrame(pdf)

# COMMAND ----------

df.show()

# COMMAND ----------

# Define a Pandas UDF 'multiply_two_cols'
@pandas_udf("long")
def multiply_two_cols(
        iterator: Iterator[Tuple[pd.Series, pd.Series]]) -> Iterator[pd.Series]:
    for a, b in iterator:
        yield a * b

# COMMAND ----------

# Apply the 'multiply_two_cols' UDF to the "x" column of the Spark DataFrame
df.select(multiply_two_cols("x", "x")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Series to scalar UDF

# COMMAND ----------

# MAGIC %md
# MAGIC - Similar to Spark aggregate functions
# MAGIC - Defines an aggregation from one or more pandas Series to a scalar value, where each pandas Series represents a Spark column
# MAGIC - **Use:** With APIs such as select, withColumn, groupBy.agg, and pyspark.sql.Window

# COMMAND ----------

# MAGIC %md
# MAGIC - The return type should be a primitive data type, and the returned scalar can be either a Python primitive type, **for example**, int or float or a NumPy data type (numpy.int64 or numpy.float64)
# MAGIC - Does not support partial aggregation and all data for each group is loaded into memory

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute mean with select, groupBy, and window operations

# COMMAND ----------

# Importing Libraries
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql import Window

# COMMAND ----------

# Creating DataFrame
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

# COMMAND ----------

df.show()

# COMMAND ----------

#  Define a Pandas UDF 'mean_udf'
@pandas_udf("double")
def mean_udf(v: pd.Series) -> float:
    return v.mean()

# COMMAND ----------

# Apply the 'mean_udf' UDF
df.select(mean_udf(df['v'])).show()

# COMMAND ----------

# Use UDF with GroupBy
df.groupby("id").agg(mean_udf(df['v'])).show()

# COMMAND ----------

# Using Window Function
w = Window \
    .partitionBy('id') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df.withColumn('mean_v', mean_udf(df['v']).over(w)).show()
