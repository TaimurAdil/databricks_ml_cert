# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **Resources:**
# MAGIC
# MAGIC **1) Connect to Azure Data Lake Storage Gen2 and Blob Storage -** https://learn.microsoft.com/en-us/azure/databricks/external-data/azure-storage#--access-azure-data-lake-storage-gen2-or-blob-storage-using-the-account-key
# MAGIC
# MAGIC **2) Azure Data Lake Storage Gen2 URI -** https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-abfs-driver

# COMMAND ----------

# Code from azure documentation

spark.conf.set(
    "fs.azure.account.key.<storage-account>.dfs.core.windows.net",
    dbutils.secrets.get(scope="<scope>", key="<storage-account-access-key>"))

# COMMAND ----------

# Code we will use
# Replace storage account name and access key

spark.conf.set(
    "fs.azure.account.key.gen2storage05.dfs.core.windows.net",
    "8p/uTMh15uCb5eDUYRTH/+FkZXOvfeuxsO6jYgfzQacfKOERZdmlvt9QWjdeo0E0AnYmCP3v1foJ+AStPJpn/A==")

# COMMAND ----------

# URI

abfs[s]://file_system@account_name.dfs.core.windows.net/<path>/<path>/<file_name>

# COMMAND ----------

bank_data = spark.read.csv("abfss://input@gen2storage05.dfs.core.windows.net/bank_data.csv", header=True)

# COMMAND ----------

display(bank_data)
