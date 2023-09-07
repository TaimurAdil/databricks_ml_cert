# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Store data with Pandas

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df = pd.DataFrame([["a", 1], ["b", 2], ["c", 3]])
print(df)

# COMMAND ----------

df.to_csv("/dbfs/dbfs_test.csv")

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

df.to_csv("./relative_path_test.csv")

# COMMAND ----------

# MAGIC %sh ls

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Load data with Pandas

# COMMAND ----------

df_1 = pd.read_csv("/dbfs/dbfs_test.csv")
df_2 = pd.read_csv("./relative_path_test.csv")

# COMMAND ----------

print(df_1)

# COMMAND ----------

print(df_2)
