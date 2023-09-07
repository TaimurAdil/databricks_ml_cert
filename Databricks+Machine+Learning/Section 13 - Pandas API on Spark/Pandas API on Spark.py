# Databricks notebook source
# MAGIC %md
# MAGIC # Resources
# MAGIC **1) Pandas API on Spark:** https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html
# MAGIC
# MAGIC **2) Pandas API on Spark:** https://api-docs.databricks.com/python/pyspark/latest/pyspark.pandas/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 1: Migration from pandas to pandas API on Spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object creation - Series

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

# Create pandas series
pser = pd.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

print(pser)

# COMMAND ----------

type(pser)

# COMMAND ----------

# Create pandas-on-spark series
psser = ps.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

print(psser)

# COMMAND ----------

type(psser)

# COMMAND ----------

# Create a pandas-on-spark series by passing a pandas series
psser_1 = ps.Series(pser)

# COMMAND ----------

psser_1

# COMMAND ----------

type(psser_1)

# COMMAND ----------

# Create a pandas-on-spark series by passing a pandas series
psser_2 = ps.from_pandas(pser)

# COMMAND ----------

psser_2

# COMMAND ----------

type(psser_2)

# COMMAND ----------

# sort_index method
print(psser_1.sort_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Object creation - Dataframe

# COMMAND ----------

my_dict = {"A": np.random.rand(5),
           "B": np.random.rand(5)}

# COMMAND ----------

my_dict

# COMMAND ----------

# Create a pandas dataframe
pdf = pd.DataFrame(my_dict)

# COMMAND ----------

pdf

# COMMAND ----------

type(pdf)

# COMMAND ----------

# Create a pandas-on-spark dataframe
psdf = ps.DataFrame(my_dict)

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

# Create a pandas-on-spark dataframe by passing a pandas dataframe
psdf_1 = ps.DataFrame(pdf)
psdf_2 = ps.from_pandas(pdf)

# COMMAND ----------

psdf_1

# COMMAND ----------

psdf_2

# COMMAND ----------

print(type(psdf_1))
print(type(psdf_2))

# COMMAND ----------

# sort_index method
psdf_1.sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Viewing data

# COMMAND ----------

# Create pandas-on-spark series
psser = ps.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------


# Create a pandas-on-spark dataframe
psdf = ps.DataFrame(my_dict)

# COMMAND ----------

psser

# COMMAND ----------

psdf

# COMMAND ----------

psser.head(2)

# COMMAND ----------

psdf.head(3)

# COMMAND ----------

# Summary statistics
psser.describe()

# COMMAND ----------

# Summary statistics
psdf.describe()

# COMMAND ----------

# Sort values method
psser.sort_values()

# COMMAND ----------

psdf

# COMMAND ----------

# Sort values method
psdf.sort_values(by="A")

# COMMAND ----------

psdf

# COMMAND ----------

# Transpose method
psdf.transpose()

# COMMAND ----------

# Transpose method
psser.transpose()

# COMMAND ----------

ps.get_option('compute.max_rows')

# COMMAND ----------

ps.set_option('compute.max_rows', 2000)

# COMMAND ----------

ps.get_option('compute.max_rows')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selection

# COMMAND ----------

# Create pandas-on-spark series
psser = ps.Series([1, 3, 5, np.nan, 6, 8])

# COMMAND ----------

# Create a pandas-on-spark dataframe
psdf = ps.DataFrame(my_dict)

# COMMAND ----------

psser

# COMMAND ----------

psdf

# COMMAND ----------

psdf['A']

# COMMAND ----------

psdf['B']

# COMMAND ----------

psdf[['A', 'B']]

# COMMAND ----------

psdf.B

# COMMAND ----------

psdf

# COMMAND ----------

psdf.loc[0:3]

# COMMAND ----------

psser

# COMMAND ----------

psser.loc[0:2]

# COMMAND ----------

psser.loc[4:5]

# COMMAND ----------

psdf

# COMMAND ----------

# Slicing
psdf.iloc[0:5, 0:2]

# COMMAND ----------

psdf.iloc[0:3, 0:2]

# COMMAND ----------

psser = ps.Series([100, 200, 300, 400, 500], index=[0, 1, 2, 3, 4])

# COMMAND ----------

psser

# COMMAND ----------

psdf["C"] = psser

# COMMAND ----------

# Those are needed for managing options
from pyspark.pandas.config import set_option, reset_option
set_option("compute.ops_on_diff_frames", True)
psdf['C'] = psser
 
# Reset to default to avoid potential expensive operation in the future
reset_option("compute.ops_on_diff_frames")
print(psdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Applying Python function with pandas-on-Spark object

# COMMAND ----------

psdf

# COMMAND ----------

psdf.apply(np.cumsum)

# COMMAND ----------

psdf

# COMMAND ----------

psdf.apply(np.cumsum, axis=1)

# COMMAND ----------

psdf.apply(lambda x: x ** 2)

# COMMAND ----------

def square(x) -> ps.Series[np.float64]:
    return x ** 2

# COMMAND ----------

psdf.apply(square)

# COMMAND ----------

psdf_5 = ps.DataFrame({"A": range(1000)})

# COMMAND ----------

print(psdf_5)

# COMMAND ----------

len(psdf_5)

# COMMAND ----------

# Working properly since size of data <= compute.shortcut_limit (1000)
ps.DataFrame({'A': range(1000)}).apply(lambda col: col.max())

# COMMAND ----------

# Not working properly since size of data > compute.shortcut_limit (1000)
ps.DataFrame({'A': range(1200)}).apply(lambda col: col.max())

# COMMAND ----------

# Set compute.shortcut_limit = 1200
ps.set_option('compute.shortcut_limit', 1200)

# COMMAND ----------

# Not working properly since size of data > compute.shortcut_limit (1000)
ps.DataFrame({'A': range(1200)}).apply(lambda col: col.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grouping Data

# COMMAND ----------

# Create a pandas-on-Spark DataFrame
psdf = ps.DataFrame({'A': [1, 2, 2, 3, 4],
                    'B': [10, 20, 30, 30, 50],
                    'C': [5, 7, 9, 11, 13]})

# COMMAND ----------

psdf

# COMMAND ----------

psdf.groupby("A").sum()

# COMMAND ----------

psdf.groupby(["A", "B"]).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plotting

# COMMAND ----------

# This is needed for visualizing plot on notebook
%matplotlib inline

# COMMAND ----------

# bar plot

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
         
psdf = ps.DataFrame({'speed': speed,
                     'lifespan': lifespan}, index=index)
psdf.plot.bar()

# COMMAND ----------

psdf

# COMMAND ----------

# horizontal bar plot
psdf.plot.barh()

# COMMAND ----------

#  pie chart

psdf = ps.DataFrame({'mass': [0.330, 4.87, 5.97],
                     'radius': [2439.7, 6051.8, 6378.1]},
                    index=['Mercury', 'Venus', 'Earth'])
psdf.plot.pie(y='mass')

# COMMAND ----------

# area plot

psdf = ps.DataFrame({
    'sales': [3, 2, 3, 9, 10, 6, 3],
    'signups': [5, 5, 6, 12, 14, 13, 9],
    'visits': [20, 42, 28, 62, 81, 50, 90],
}, index=pd.date_range(start='2019/08/15', end='2020/03/09',
                       freq='M'))
psdf.plot.area()

# COMMAND ----------

# line plot

psdf = ps.DataFrame({'rabbit': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]},
                    index=[1990, 1997, 2003, 2009, 2014])
psdf.plot.line()

# COMMAND ----------

# Histogram

pdf = pd.DataFrame(
    np.random.randint(1, 7, 6000),
    columns=['one'])
pdf['two'] = pdf['one'] + np.random.randint(1, 7, 6000)
psdf = ps.from_pandas(pdf)
psdf.plot.hist(bins=12, alpha=0.5)

# COMMAND ----------

# scatter plot

psdf = ps.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])
psdf.plot.scatter(x='length',
                  y='width',
                  c='species')

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Missing Functionalities and Workarounds in pandas API on Spark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Directly use pandas APIs through type conversion

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

psdf = ps.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
                   columns=['length', 'width', 'species'])

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

psidx = psdf.index

# COMMAND ----------

psidx

# COMMAND ----------

type(psidx)

# COMMAND ----------

ps_list = psidx.to_list()

# COMMAND ----------

ps_list

# COMMAND ----------

type(ps_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Native Support for pandas Objects

# COMMAND ----------

psdf = ps.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3] * 4, dtype='int32'),
                     'F': 'foo'})

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed execution for pandas functions

# COMMAND ----------

i = pd.date_range('2018-04-09', periods=2000, freq='1D1min')
ts = ps.DataFrame({'A': ['timestamp']}, index=i)

# COMMAND ----------

print(ts)

# COMMAND ----------

len(ts)

# COMMAND ----------

ts.between_time('0:15', '0:16')

# COMMAND ----------

ts.to_pandas().between_time('0:15', '0:16')

# COMMAND ----------

ts.pandas_on_spark.apply_batch(func=lambda pdf: pdf.between_time('0:15', '0:16'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using SQL in pandas API on Spark

# COMMAND ----------

psdf = ps.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                     'rabbit': [20, 18, 489, 675, 1776],
                     'horse': [4, 25, 281, 600, 1900]})

# COMMAND ----------

pdf = pd.DataFrame({'year': [1990, 1997, 2003, 2009, 2014],
                    'sheep': [22, 50, 121, 445, 791],
                    'chicken': [250, 326, 589, 1241, 2118]})

# COMMAND ----------

psdf

# COMMAND ----------

pdf

# COMMAND ----------

ps.sql("SELECT * FROM {psdf} WHERE rabbit > 100", psdf = psdf)

# COMMAND ----------

ps.sql('''
    SELECT ps.rabbit, pd.chicken
    FROM {psdf} ps INNER JOIN {pdf} pd
    ON ps.year = pd.year
    ORDER BY ps.rabbit, pd.chicken''', psdf=psdf, pdf=pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 3: Working with PySpark

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conversion from and to PySpark DataFrame

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark.pandas as ps

# COMMAND ----------

# Creating a pandas-on-spark DataFrame
psdf = ps.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

# COMMAND ----------

psdf

# COMMAND ----------

type(psdf)

# COMMAND ----------

# Converting pandas-on-spark DataFrame to Spark DataFrame
sdf = psdf.to_spark()

# COMMAND ----------

sdf

# COMMAND ----------

sdf.show()

# COMMAND ----------

type(sdf)

# COMMAND ----------

psdf_2 = sdf.to_pandas_on_spark()

# COMMAND ----------

type(psdf_2)

# COMMAND ----------

psdf_3 = sdf.pandas_api()

# COMMAND ----------

type(psdf_3)

# COMMAND ----------

psdf_3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking Spark execution plans

# COMMAND ----------

from pyspark.pandas import option_context

with option_context(
        "compute.ops_on_diff_frames", True,
        "compute.default_index_type", 'distributed'):
    df = ps.range(10) + ps.range(10)
    df.spark.explain()

# COMMAND ----------

with option_context(
        "compute.ops_on_diff_frames", False,
        "compute.default_index_type", 'distributed'):
    df = ps.range(10)
    df = df + df
    df.spark.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caching DataFrames

# COMMAND ----------

with option_context("compute.default_index_type", 'distributed'):
    df = ps.range(10)
    new_df = (df + df).spark.cache()  # `(df + df)` is cached here as `df`
    new_df.spark.explain()

# COMMAND ----------

new_df.spark.unpersist()

# COMMAND ----------

with (df + df).spark.cache() as df:
    df.spark.explain()
